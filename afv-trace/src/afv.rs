//! The AFV SQLite container (`afv.export_afv`), written row for row as the reference does
//!
//! Every stored value matches the reference: the schema text (SQLite keeps it verbatim in
//! `sqlite_master`), the metadata JSON with Python's float repr and key order, the fibre rows, the
//! little-endian float64 blocks and the R*Tree rows. Only the per-run uuid and SQLite's page layout
//! differ, so outputs are compared table by table rather than by file hash

use std::fs::{self, File};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, bail, eyre};
use rusqlite::{Connection, params};
use serde_json::{Map, Value, json};

use crate::pyfmt::{np_sum_f64, py_json};

const SCHEMA: &str = "
PRAGMA application_id=0x56434643;
PRAGMA user_version=1;
CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE fibers(id INTEGER PRIMARY KEY,name TEXT NOT NULL,family TEXT NOT NULL,
 point_count INTEGER NOT NULL CHECK(point_count>=2),length REAL NOT NULL,
 min_x REAL NOT NULL,max_x REAL NOT NULL,min_y REAL NOT NULL,max_y REAL NOT NULL,
 min_z REAL NOT NULL,max_z REAL NOT NULL,annotation TEXT NOT NULL);
CREATE TABLE blocks(id INTEGER PRIMARY KEY,fiber_id INTEGER NOT NULL REFERENCES fibers(id),
 first_segment INTEGER NOT NULL,points BLOB NOT NULL,UNIQUE(fiber_id,first_segment));
CREATE VIRTUAL TABLE block_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);
";

/// What the export needs beyond the curves
pub struct ExportOptions<'a> {
    pub voxel_um: f64,
    pub origin_xyz: [i64; 3],
    pub volume_shape: [i64; 3],
    /// `--min-length-mm`; the reference stores the untouched default as the integer 0
    pub minimum_mm: MinLength,
    pub coordinate_space: Option<&'a str>,
}

/// `--min-length-mm` as argparse leaves it: the integer default or a parsed float
#[derive(Clone, Copy, Debug)]
pub enum MinLength {
    Default,
    Given(f64),
}

impl MinLength {
    pub fn value(self) -> f64 {
        match self {
            MinLength::Default => 0.0,
            MinLength::Given(v) => v,
        }
    }

    fn json(self) -> Value {
        match self {
            MinLength::Default => json!(0),
            MinLength::Given(v) => json!(v),
        }
    }
}

/// Totals stored in the finished file's metadata
#[derive(Debug, Default)]
pub struct Summary {
    pub fiber_count: i64,
    pub point_count: i64,
    pub block_count: i64,
}

/// Fill `buf` (at most 256 bytes) from the kernel's random source
///
/// `getentropy` rather than `getrandom`, since macOS has only the former; glibc has had both since 2.25
fn fill_random(buf: &mut [u8]) -> Result<()> {
    // SAFETY: the pointer and length describe `buf`, which getentropy fills in full or not at all
    if unsafe { libc::getentropy(buf.as_mut_ptr().cast(), buf.len()) } != 0 {
        bail!("getentropy failed: {}", std::io::Error::last_os_error());
    }
    Ok(())
}

/// A random version 4 UUID in canonical form, from the kernel's random source
fn uuid4() -> Result<String> {
    let mut b = [0u8; 16];
    fill_random(&mut b)?;
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let h: String = b.iter().map(|v| format!("{v:02x}")).collect();
    Ok(format!("{}-{}-{}-{}-{}", &h[..8], &h[8..12], &h[12..16], &h[16..20], &h[20..]))
}

/// `tempfile.mkstemp(prefix=".afv-", suffix=".partial", dir=parent)`: an exclusively created, empty file
fn mkstemp(parent: &Path) -> Result<PathBuf> {
    for _ in 0..100 {
        let mut b = [0u8; 6];
        fill_random(&mut b)?;
        const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789_";
        let name: String = b.iter().map(|v| ALPHABET[*v as usize % ALPHABET.len()] as char).collect();
        let path = parent.join(format!(".afv-{name}.partial"));
        match fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path) {
            Ok(_) => return Ok(path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    bail!("No usable temporary file name found in {}", parent.display())
}

/// `bounds(points)`: min x, max x, min y, max y, min z, max z
fn bounds(q: &[[f64; 3]]) -> [f64; 6] {
    let mut out =
        [f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY];
    for p in q {
        for a in 0..3 {
            out[2 * a] = out[2 * a].min(p[a]);
            out[2 * a + 1] = out[2 * a + 1].max(p[a]);
        }
    }
    out
}

/// Removes the temporary file however the export ends, as the reference's `finally` does
struct TempGuard(PathBuf);

impl Drop for TempGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Write `curves` (family 1 = V or 2 = H, tracing-space float32 x, y, z points) to `destination`
pub fn export_afv<I>(curves: I, destination: &Path, opt: &ExportOptions) -> Result<Summary>
where
    I: IntoIterator<Item = Result<(usize, Vec<[f32; 3]>)>>,
{
    if destination.exists() {
        bail!("Output already exists: {}", destination.display());
    }
    let parent = destination.parent().map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    let parent = if parent.as_os_str().is_empty() { PathBuf::from(".") } else { parent };
    fs::create_dir_all(&parent)?;
    let temporary = mkstemp(&parent)?;
    let _guard = TempGuard(temporary.clone());
    let summary = write_db(curves, &temporary, opt)?;
    File::open(&temporary)?.sync_all()?;
    // a hard link publishes the complete file without overwriting a path that another process may
    // have created while tracing was running
    fs::hard_link(&temporary, destination).map_err(|e| eyre!("{}: {e}", destination.display()))?;
    Ok(summary)
}

fn metadata(opt: &ExportOptions) -> Result<Vec<(&'static str, Value)>> {
    let mut frame = Map::new();
    frame.insert("coordinate_base_shape_zyx".into(), json!(opt.volume_shape));
    frame.insert("vc_open_data_source_coordinate_level".into(), json!(0));
    frame.insert("vc_open_data_source_coordinate_scale_factor".into(), json!(1));
    frame.insert("vc_open_data_source_original_resolution".into(), json!(opt.voxel_um));
    if let Some(space) = opt.coordinate_space.filter(|s| !s.is_empty()) {
        frame.insert("vc_open_data_coordinate_space".into(), json!(space));
    }
    let mut root = Map::new();
    root.insert("type".into(), json!("vc3d_fiber_collection"));
    root.insert("version".into(), json!(1));
    root.insert("scale".into(), json!(1));
    root.extend(frame.clone());
    root.insert("coordinate_order".into(), json!("XYZ"));
    root.insert("coordinate_units".into(), json!("voxel"));
    root.insert("voxel_size_mm".into(), json!(opt.voxel_um / 1000.0));
    let mut tracing = Map::new();
    tracing.insert("scale".into(), json!(2));
    tracing.insert("center_offset".into(), json!(0.5));
    tracing.insert("origin_xyz".into(), json!(opt.origin_xyz));
    tracing.insert("minimum_length_mm".into(), opt.minimum_mm.json());
    tracing.insert("family_locked".into(), json!(true));
    tracing.insert("ct_exterior_filter".into(), json!(false));
    Ok(vec![
        ("complete", json!(false)),
        ("uuid", json!(uuid4()?)),
        ("root", Value::Object(root)),
        ("frame", Value::Object(frame)),
        ("geometry", json!("XYZ binary64 little-endian native L0")),
        ("block_segments", json!(256)),
        ("source", json!("AVF_SPLINE_GENERATOR")),
        ("tracing", Value::Object(tracing)),
    ])
}

fn write_db<I>(curves: I, path: &Path, opt: &ExportOptions) -> Result<Summary>
where
    I: IntoIterator<Item = Result<(usize, Vec<[f32; 3]>)>>,
{
    let mut db = Connection::open(path)?;
    db.execute_batch("PRAGMA foreign_keys=ON")?;
    db.execute_batch("PRAGMA cache_size=-32768")?;
    db.execute_batch(SCHEMA)?;
    let tx = db.transaction()?;
    {
        let mut meta = tx.prepare("INSERT INTO metadata VALUES(?,?)")?;
        for (k, v) in metadata(opt)? {
            meta.execute(params![k, py_json(&v)])?;
        }
    }
    let origin = opt.origin_xyz.map(|v| v as f64);
    let limit = [opt.volume_shape[2], opt.volume_shape[1], opt.volume_shape[0]].map(|v| v as f64);
    let mut s = Summary::default();
    {
        let mut fiber = tx.prepare("INSERT INTO fibers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)")?;
        let mut block = tx.prepare("INSERT INTO blocks VALUES(?,?,?,?)")?;
        let mut bound = tx.prepare("INSERT INTO block_bounds VALUES(?,?,?,?,?,?,?)")?;
        let mut q: Vec<[f64; 3]> = Vec::new();
        let mut seg: Vec<f64> = Vec::new();
        for item in curves {
            let (family, points) = item?;
            // points.astype("<f8") * 2 + 0.5 + origin
            q.clear();
            q.extend(points.iter().map(|p| [0, 1, 2].map(|a| p[a] as f64 * 2.0 + 0.5 + origin[a])));
            let bad = q.iter().any(|p| (0..3).any(|a| !p[a].is_finite() || p[a] < 0.0 || p[a] >= limit[a]));
            if q.len() < 2 || bad {
                bail!("Spline contains invalid native coordinates");
            }
            // np.linalg.norm(np.diff(q, axis=0), axis=1).sum()
            seg.clear();
            seg.extend(q.windows(2).map(|w| {
                let d = [w[1][0] - w[0][0], w[1][1] - w[0][1], w[1][2] - w[0][2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            }));
            let length = np_sum_f64(&seg);
            let mm = length * opt.voxel_um / 1000.0;
            if length <= 0.0 || mm < opt.minimum_mm.value() {
                continue;
            }
            s.fiber_count += 1;
            let count = s.fiber_count;
            let label = match family {
                1 => "V",
                2 => "H",
                other => bail!("Invalid family {other}"),
            };
            let name = format!("fiber_{count}.json");
            let annotation = json!({
                "type": "vc3d_fiber",
                "version": 1,
                "sequence": count,
                "filename": name,
                "hv_classification": {"manual_tag": label},
                "classification_source": "model_prediction_family_locked_tracing",
                "control_points": [q[0], q[q.len() - 1]],
                "length_voxels": length,
                "length_mm": mm,
            });
            let b = bounds(&q);
            fiber.execute(params![
                count,
                name,
                label,
                q.len() as i64,
                length,
                b[0],
                b[1],
                b[2],
                b[3],
                b[4],
                b[5],
                py_json(&annotation)
            ])?;
            for first in (0..q.len() - 1).step_by(256) {
                let pts = &q[first..(first + 257).min(q.len())];
                let bytes: Vec<u8> = pts.iter().flat_map(|p| p.iter().flat_map(|v| v.to_le_bytes())).collect();
                s.block_count += 1;
                block.execute(params![s.block_count, count, first as i64, bytes])?;
                let b = bounds(pts);
                bound.execute(params![s.block_count, b[0], b[1], b[2], b[3], b[4], b[5]])?;
            }
            s.point_count += q.len() as i64;
        }
    }
    tx.execute_batch("CREATE INDEX fibers_length ON fibers(length DESC,id)")?;
    {
        let mut meta = tx.prepare("INSERT OR REPLACE INTO metadata VALUES(?,?)")?;
        let rows = [
            ("complete", json!(true)),
            ("fiber_count", json!(s.fiber_count)),
            ("point_count", json!(s.point_count)),
            ("block_count", json!(s.block_count)),
        ];
        for (k, v) in rows {
            meta.execute(params![k, py_json(&v)])?;
        }
    }
    tx.commit()?;
    check(&db)?;
    db.close().map_err(|(_, e)| e)?;
    Ok(s)
}

fn check(db: &Connection) -> Result<()> {
    let integrity: Vec<String> =
        db.prepare("PRAGMA integrity_check")?.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?;
    if integrity != ["ok"] {
        bail!("AFV integrity check failed");
    }
    if db.prepare("PRAGMA foreign_key_check")?.query([])?.next()?.is_some() {
        bail!("AFV contains invalid block references");
    }
    let rtree: String = db.query_row("SELECT rtreecheck('block_bounds')", [], |r| r.get(0))?;
    if rtree != "ok" {
        bail!("AFV spatial index check failed");
    }
    Ok(())
}
