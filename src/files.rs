//! Photo discovery, absen parsing, deduplication, and 4R pairing.

use std::collections::HashMap;
use std::fs::read_dir;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};

/// Represents an input class directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassDir {
    pub name: String,
    pub path: PathBuf,
}

/// Represents a validated student photo sorted by absen number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Photo {
    pub number: u32,
    pub label: String,
    pub path: PathBuf,
    pub stem: String,
}

/// Records a file skipped during discovery along with the reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skipped {
    pub path: PathBuf,
    pub filename: String,
    pub reason: String,
}

/// Extracts the class name from a directory path.
/// Uses the last directory name (file_name), with fallback to canonicalize().
/// If the folder name cannot be determined (e.g. root or empty), falls back to "folder input".
pub fn extract_folder_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
        .or_else(|| {
            path.canonicalize().ok().and_then(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| s.to_string())
            })
        })
        .unwrap_or_else(|| "folder input".to_string())
}

/// Collects class folders from the input directory.
/// If input has subdirectories, each is treated as a class, sorted case-insensitively.
/// If input has no subdirectories, processes the input folder itself with its folder name as the class name.
pub fn collect_classes(input: &Path) -> Result<Vec<ClassDir>> {
    let mut subdirs = Vec::new();

    let entries = read_dir(input)
        .with_context(|| format!("Gagal membaca folder input: {}", input.display()))?;

    for entry in entries {
        let entry =
            entry.with_context(|| format!("Gagal membaca entri di folder: {}", input.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("Gagal membaca tipe file: {}", entry.path().display()))?;
        if file_type.is_dir() {
            let name = entry.file_name().to_string_lossy().into_owned();
            subdirs.push(ClassDir {
                name,
                path: entry.path(),
            });
        }
    }

    if subdirs.is_empty() {
        let name = extract_folder_name(input);
        Ok(vec![ClassDir {
            name,
            path: input.to_path_buf(),
        }])
    } else {
        subdirs.sort_by_key(|a| a.name.to_lowercase());
        Ok(subdirs)
    }
}

/// Collects photos in a class folder, port of `kumpulkan_foto`.
///
/// Filters out non-photo files silently, skips processed/unindexed files with reasons,
/// deduplicates identical absen numbers favoring the newer file, and sorts by absen number.
pub fn collect_photos(dir: &Path) -> Result<(Vec<Photo>, Vec<Skipped>)> {
    let mut file_entries = Vec::new();

    let entries =
        read_dir(dir).with_context(|| format!("Gagal membaca folder kelas: {}", dir.display()))?;

    for entry in entries {
        let entry =
            entry.with_context(|| format!("Gagal membaca entri di folder: {}", dir.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("Gagal membaca tipe file: {}", entry.path().display()))?;
        if file_type.is_file() {
            file_entries.push(entry);
        }
    }

    // Sort entries case-insensitively by filename, matching reference:
    // sorted((e for e in it if e.is_file()), key=lambda e: e.name.lower())
    file_entries.sort_by_key(|a| a.file_name().to_string_lossy().to_lowercase());

    let mut skipped = Vec::new();
    let mut per_number: HashMap<u32, (Photo, SystemTime)> = HashMap::new();

    for entry in file_entries {
        let path = entry.path();
        let filename = entry.file_name().to_string_lossy().into_owned();

        // Silently skip files starting with . or ~
        if filename.starts_with('.') || filename.starts_with('~') {
            continue;
        }

        // Silently skip non-photo extensions (.jpg, .jpeg, .png)
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();
        if ext != "jpg" && ext != "jpeg" && ext != "png" {
            continue;
        }

        // File stem without extension
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        let stem_lower = stem.to_lowercase();
        // Skip processed files ending in _3x4 or _4r
        if stem_lower.ends_with("_3x4") || stem_lower.ends_with("_4r") {
            skipped.push(Skipped {
                path,
                filename,
                reason: "dilewati: ini file hasil olahan, bukan foto mentah".to_string(),
            });
            continue;
        }

        // Match leading digits with optional leading whitespace: r'^\s*(\d+)'
        let trimmed = stem.trim_start();
        let digits: String = trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            skipped.push(Skipped {
                path,
                filename,
                reason: "dilewati: nama file tidak diawali nomor absen".to_string(),
            });
            continue;
        }

        // Parse absen number as u32 (handles integer overflow gracefully)
        let number = match digits.parse::<u32>() {
            Ok(n) => n,
            Err(_) => {
                skipped.push(Skipped {
                    path,
                    filename,
                    reason: "dilewati: nomor absen tidak valid (terlalu besar)".to_string(),
                });
                continue;
            }
        };

        let label = digits;
        let mtime = match entry.metadata().and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(e) => {
                return Err(e).with_context(|| {
                    format!("Gagal membaca waktu modifikasi file: {}", path.display())
                });
            }
        };

        let photo = Photo {
            number,
            label,
            path: path.clone(),
            stem,
        };

        if let Some((old_photo, old_mtime)) = per_number.get(&number) {
            // Reference: baru, dibuang = (foto, lama) if foto["mtime"] > lama["mtime"] else (lama, foto)
            // Strictly greater than: if equal, old_photo is retained
            if mtime > *old_mtime {
                let dibuang_name = old_photo
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| old_photo.stem.clone());
                let baru_name = filename;
                skipped.push(Skipped {
                    path: old_photo.path.clone(),
                    filename: dibuang_name,
                    reason: format!(
                        "dilewati: nomor absen {} dobel, yang dipakai {}",
                        number, baru_name
                    ),
                });
                per_number.insert(number, (photo, mtime));
            } else {
                let dibuang_name = filename;
                let baru_name = old_photo
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| old_photo.stem.clone());
                skipped.push(Skipped {
                    path,
                    filename: dibuang_name,
                    reason: format!(
                        "dilewati: nomor absen {} dobel, yang dipakai {}",
                        number, baru_name
                    ),
                });
            }
        } else {
            per_number.insert(number, (photo, mtime));
        }
    }

    let mut photos: Vec<Photo> = per_number.into_values().map(|(p, _)| p).collect();
    photos.sort_by_key(|p| p.number);

    Ok((photos, skipped))
}

/// Pairs photos for 4R sheet printing: [(0, Some(1)), (2, Some(3)), ...].
/// An odd number of photos leaves the last pair with None for the second photo.
pub fn pair_photos(photos: &[Photo]) -> Vec<(Photo, Option<Photo>)> {
    let mut pairs = Vec::with_capacity(photos.len().div_ceil(2));
    let mut i = 0;
    while i < photos.len() {
        let first = photos[i].clone();
        let second = if i + 1 < photos.len() {
            Some(photos[i + 1].clone())
        } else {
            None
        };
        pairs.push((first, second));
        i += 2;
    }
    pairs
}

/// Generates a summary string for 4R photo pairing, e.g. "1&2, 3&4, 5" or "1&2, 3&4, 5&10, 11&20".
pub fn pairing_summary(pairs: &[(Photo, Option<Photo>)]) -> String {
    pairs
        .iter()
        .map(|(first, second)| match second {
            Some(second) => format!("{}&{}", first.label, second.label),
            None => first.label.clone(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{File, create_dir, write};
    use std::time::Duration;
    use tempfile::tempdir;

    #[test]
    fn test_collect_classes() {
        let dir = tempdir().expect("create temp dir");
        let root = dir.path();

        // 1. With subdirectories
        create_dir(root.join("1A")).expect("create 1A");
        create_dir(root.join("X-1")).expect("create X-1");
        create_dir(root.join("2B")).expect("create 2B");
        write(root.join("desktop.ini"), b"").expect("write loose file");

        let classes = collect_classes(root).expect("collect classes");
        let names: Vec<String> = classes.into_iter().map(|c| c.name).collect();
        assert_eq!(names, vec!["1A", "2B", "X-1"]);

        // 2. Without subdirectories
        let single_dir = root.join("1A");
        let single_classes = collect_classes(&single_dir).expect("collect single class");
        assert_eq!(single_classes.len(), 1);
        assert_eq!(single_classes[0].name, "1A");
        assert_eq!(single_classes[0].path, single_dir);
    }

    #[test]
    fn test_extract_folder_name() {
        assert_eq!(extract_folder_name(Path::new("Input_Raw/X-1")), "X-1");
        assert_eq!(extract_folder_name(Path::new("Input_Raw\\X-1")), "X-1");
        assert_eq!(extract_folder_name(Path::new("testdata/raw")), "raw");
        assert_eq!(extract_folder_name(Path::new("")), "folder input");
    }

    #[test]
    fn test_collect_photos_and_pairing() {
        let dir = tempdir().expect("create temp dir");
        let folder = dir.path().join("kelas1");
        create_dir(&folder).expect("create class folder");

        // Create files according to PRD G1.2 specification 5
        let files = [
            "1_A.jpg",
            "2_B.JPG",
            "3_C.jpg",
            "4_D.jpg",
            "4_D (ulang).jpg",
            "5_E.jpeg",
            "10_F.png",
            "11_G.jpg",
            "20_H.jpg",
            "1_A_3x4.jpg",
            "IMG_1234.jpg",
            "desktop.ini",
        ];

        for name in &files {
            write(folder.join(name), b"dummy").expect("write test file");
        }

        // Set 4_D (ulang).jpg to a strictly newer mtime
        let ulang_path = folder.join("4_D (ulang).jpg");
        let file = File::options()
            .write(true)
            .open(&ulang_path)
            .expect("open ulang file");
        file.set_modified(SystemTime::now() + Duration::from_secs(10))
            .expect("set modified time");

        let (photos, skipped) = collect_photos(&folder).expect("collect photos");

        // Verify photos and numbers
        let numbers: Vec<u32> = photos.iter().map(|p| p.number).collect();
        assert_eq!(numbers, vec![1, 2, 3, 4, 5, 10, 11, 20]);

        // Verify photo 4 is 4_D (ulang)
        let p4 = photos
            .iter()
            .find(|p| p.number == 4)
            .expect("photo 4 exists");
        assert_eq!(p4.stem, "4_D (ulang)");

        // Verify skipped: exactly 3 items
        assert_eq!(skipped.len(), 3);

        let skipped_dup = skipped.iter().find(|s| s.filename == "4_D.jpg");
        assert!(skipped_dup.is_some());
        assert_eq!(
            skipped_dup.unwrap().reason,
            "dilewati: nomor absen 4 dobel, yang dipakai 4_D (ulang).jpg"
        );

        let skipped_3x4 = skipped.iter().find(|s| s.filename == "1_A_3x4.jpg");
        assert!(skipped_3x4.is_some());
        assert_eq!(
            skipped_3x4.unwrap().reason,
            "dilewati: ini file hasil olahan, bukan foto mentah"
        );

        let skipped_no_num = skipped.iter().find(|s| s.filename == "IMG_1234.jpg");
        assert!(skipped_no_num.is_some());
        assert_eq!(
            skipped_no_num.unwrap().reason,
            "dilewati: nama file tidak diawali nomor absen"
        );

        // Verify desktop.ini is ignored completely
        assert!(!photos.iter().any(|p| p.stem == "desktop"));
        assert!(!skipped.iter().any(|s| s.filename == "desktop.ini"));

        // Verify pairing
        let pairs = pair_photos(&photos);
        assert_eq!(pairs.len(), 4);
        assert_eq!(pairs[0].0.number, 1);
        assert_eq!(pairs[0].1.as_ref().unwrap().number, 2);
        assert_eq!(pairs[1].0.number, 3);
        assert_eq!(pairs[1].1.as_ref().unwrap().stem, "4_D (ulang)");
        assert_eq!(pairs[2].0.number, 5);
        assert_eq!(pairs[2].1.as_ref().unwrap().number, 10);
        assert_eq!(pairs[3].0.number, 11);
        assert_eq!(pairs[3].1.as_ref().unwrap().number, 20);

        let summary = pairing_summary(&pairs);
        assert_eq!(summary, "1&2, 3&4, 5&10, 11&20");
    }

    #[test]
    fn test_pairing_summary_odd() {
        let photos = vec![
            Photo {
                number: 1,
                label: "1".into(),
                path: PathBuf::new(),
                stem: "1_A".into(),
            },
            Photo {
                number: 2,
                label: "2".into(),
                path: PathBuf::new(),
                stem: "2_B".into(),
            },
            Photo {
                number: 3,
                label: "3".into(),
                path: PathBuf::new(),
                stem: "3_C".into(),
            },
            Photo {
                number: 4,
                label: "4".into(),
                path: PathBuf::new(),
                stem: "4_D".into(),
            },
            Photo {
                number: 5,
                label: "5".into(),
                path: PathBuf::new(),
                stem: "5_E".into(),
            },
        ];

        let pairs = pair_photos(&photos);
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[2].0.number, 5);
        assert!(pairs[2].1.is_none());

        let summary = pairing_summary(&pairs);
        assert_eq!(summary, "1&2, 3&4, 5");
    }

    #[test]
    fn test_collect_photos_zero_padded_in_separate_folder() {
        let dir = tempdir().expect("create temp dir");
        let folder = dir.path().join("kelas2");
        create_dir(&folder).expect("create folder");

        write(folder.join("04_X.jpg"), b"dummy").expect("write test file");

        let (photos, skipped) = collect_photos(&folder).expect("collect photos");
        assert_eq!(photos.len(), 1);
        assert!(skipped.is_empty());
        assert_eq!(photos[0].number, 4);
        assert_eq!(photos[0].label, "04");
        assert_eq!(photos[0].stem, "04_X");
    }

    #[test]
    fn test_collect_photos_number_overflow() {
        let dir = tempdir().expect("create temp dir");
        let folder = dir.path().join("overflow");
        create_dir(&folder).expect("create folder");

        write(folder.join("99999999999999999999_Foto.jpg"), b"dummy").expect("write overflow file");

        let (photos, skipped) = collect_photos(&folder).expect("collect photos");
        assert!(photos.is_empty());
        assert_eq!(skipped.len(), 1);
        assert_eq!(
            skipped[0].reason,
            "dilewati: nomor absen tidak valid (terlalu besar)"
        );
    }
}
