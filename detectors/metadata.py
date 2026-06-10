from pathlib import Path

from core.types import Detection
from detectors.base import Context, Detector

EDITOR_SOFTWARE = {
    'photoshop', 'gimp', 'paint', 'paintshop',
    'illustrator', 'inkscape', 'affinity', 'pixelmator'
}

def read_exif(path: str) -> dict:
    try:
        import exifread
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        return {str(k): str(v) for k, v in tags.items()}
    except Exception:
        return {}


def read_pdf_structure(path: str) -> dict:
    try:
        import fitz
        doc  = fitz.open(path)
        meta = doc.metadata or {}
        incremental_updates = doc.xref_length() > 1
        doc.close()
        return {
            'producer':             meta.get('producer', ''),
            'creator':              meta.get('creator', ''),
            'creation_date':        meta.get('creationDate', ''),
            'mod_date':             meta.get('modDate', ''),
            'incremental_updates':  incremental_updates,
        }
    except Exception:
        return {}
    
def metadata_red_flags(meta: dict) -> tuple[float, list[str]]:
    flags = []
    score = 0.0

    for key, val in meta.items():
        if not isinstance(val, str):
            continue
        val_lower = val.lower()
        if any(sw in val_lower for sw in EDITOR_SOFTWARE):
            flags.append(f'Editing software detected: {val}')
            score += 0.4
            break

    if meta.get('incremental_updates'):
        flags.append('PDF has incremental update layers — edited after creation')
        score += 0.3

    create = meta.get('creation_date', '') or meta.get('DateTimeOriginal', '')
    modify = meta.get('mod_date', '')     or meta.get('DateTime', '')
    if create and modify and create != modify:
        flags.append(f'Modified after creation: created={create}, modified={modify}')
        score += 0.2

    return float(min(score, 1.0)), flags

class MetadataDetector(Detector):
    name = 'metadata'

    def run(self, img, ctx: Context) -> Detection:
        try:
            ext = Path(ctx.file_path).suffix.lower()
            if ext == '.pdf':
                meta = read_pdf_structure(ctx.file_path)
            else:
                meta = read_exif(ctx.file_path)

            score, flags = metadata_red_flags(meta)
            return Detection(
                detector_name=self.name,
                score=score,
                heatmap=None,
                regions=[],
                details={'flags': flags, 'raw': meta},
            )
        except Exception as e:
            d = self._empty()
            d.details['error'] = str(e)
            return d