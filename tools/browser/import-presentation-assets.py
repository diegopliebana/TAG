# Import selected image/audio bytes without redesigning or converting them.
# The manifest records where each file came from and its SHA-256 content hash.
"""Copy selected existing assets unchanged, with their licences and provenance."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import struct
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
INCOMING = ROOT / '.scratch/presentation/incoming'
DESTINATION = ROOT / 'web/assets'
SOURCES = {
    'boardgame': ('kenney_boardgame-pack.zip', 'license.txt', 'Board Game Pack v2', 'https://kenney.nl/assets/boardgame-pack'),
    'ui': ('kenney_ui-pack.zip', 'License.txt', 'UI Pack 2.0', 'https://kenney.nl/assets/ui-pack'),
    'interface': ('kenney_interface-sounds.zip', 'License.txt', 'Interface Sounds 1.0', 'https://kenney.nl/assets/interface-sounds'),
    'impact': ('kenney_impact-sounds.zip', 'License.txt', 'Impact Sounds 1.0', 'https://kenney.nl/assets/impact-sounds'),
}
SELECTION = [
    ('mark-x', 'ui', 'Vector/Blue/icon_cross.svg', 'games/tictactoe/x.svg', 'Blue X mark'),
    ('mark-o', 'boardgame', 'PNG/Pieces (Red)/pieceRed_single11.png', 'games/tictactoe/o.png', 'Red O mark'),
    ('ttt-cell', 'ui', 'Vector/Blue/button_square_border.svg', 'games/tictactoe/cell.svg', 'Repeatable cell'),
    ('c4-cell', 'ui', 'Vector/Blue/button_square_flat.svg', 'games/connect4/cell.svg', 'Repeatable blue cell'),
    ('c4-socket', 'ui', 'Vector/Blue/check_round_grey.svg', 'games/connect4/socket.svg', 'Empty socket'),
    ('c4-red', 'ui', 'Vector/Red/button_round_flat.svg', 'games/connect4/red.svg', 'Player 0 piece'),
    ('c4-yellow', 'ui', 'Vector/Yellow/button_round_flat.svg', 'games/connect4/yellow.svg', 'Player 1 piece'),
    ('sound-mark', 'interface', 'Audio/click_003.ogg', 'shared/mark.ogg', 'Mark placement candidate'),
    ('sound-land', 'boardgame', 'Bonus/chipsCollide1.ogg', 'shared/land.ogg', 'Piece landing candidate'),
    ('sound-win', 'interface', 'Audio/confirmation_001.ogg', 'shared/win.ogg', 'Win candidate'),
    ('sound-draw', 'interface', 'Audio/bong_001.ogg', 'shared/draw.ogg', 'Draw candidate'),
    ('sound-land-alternative', 'impact', 'Audio/impactSoft_medium_000.ogg', 'shared/land-alternative.ogg', 'Alternative landing cue'),
    ('sound-win-alternative', 'interface', 'Audio/confirmation_004.ogg', 'shared/win-alternative.ogg', 'Alternative win cue'),
]


def digest(data):
    """Compute a hexadecimal SHA-256 hash of data to check whether file bytes changed."""
    return hashlib.sha256(data).hexdigest()


def save(relative, data):
    """Write bytes to a checked path within the asset directory."""
    path = (DESTINATION / relative).resolve()
    path.relative_to(DESTINATION.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def run(check):
    """Copy selected archive files unchanged, or verify that files and source records match the manifest."""
    manifest = json.loads((DESTINATION / 'manifest.json').read_text(encoding='utf-8')) if check else {
        'version': 1, 'verified_at': datetime.now().strftime('%d-%m-%Y %H:%M') + ' (Europe/Paris)', 'status': 'candidate-for-review', 'author': 'Kenney / Kenney Vleugels',
        'licence': 'CC0-1.0', 'sources': {}, 'assets': [],
    }
    for source_id, (archive_name, licence_name, title, url) in SOURCES.items():
        archive = INCOMING / archive_name
        with ZipFile(archive) as package:
            licence = package.read(licence_name)
            if 'CC0' not in licence.decode('utf-8-sig'):
                raise ValueError(f'Missing CC0 evidence: {archive_name}')
            metadata = {'title': title, 'url': url, 'archive': archive_name, 'archive_sha256': digest(archive.read_bytes()),
                        'licence_path': f'licences/{source_id}.txt', 'licence_sha256': digest(licence)}
            if check:
                assert manifest['sources'][source_id] == metadata, f'Source changed: {source_id}'
                assert (DESTINATION / metadata['licence_path']).read_bytes() == licence
            else:
                manifest['sources'][source_id] = metadata
                save(metadata['licence_path'], licence)
            for asset_id, source, entry, target, role in SELECTION:
                if source != source_id:
                    continue
                data = package.read(entry)
                item = {'id': asset_id, 'source': source_id, 'entry': entry, 'path': target,
                        'role': role, 'bytes': len(data), 'sha256': digest(data),
                        'content_changes': 'none', 'presentation': 'layout, display scaling, opacity and position animation'}
                if entry.endswith('.png'):
                    item['width'], item['height'] = struct.unpack('>II', data[16:24])
                if check:
                    assert next(a for a in manifest['assets'] if a['id'] == asset_id) == item
                    assert (DESTINATION / target).read_bytes() == data, f'Asset changed: {asset_id}'
                else:
                    manifest['assets'].append(item)
                    save(target, data)
    if not check:
        save('manifest.json', (json.dumps(manifest, indent=2) + '\n').encode('utf-8'))
    print(f'{"Verified" if check else "Imported"} {len(SELECTION)} unchanged assets and four CC0 licence files.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='verify the existing manifest and original file bytes')
    run(parser.parse_args().check)
