# Presentation assets

Assets are the image and sound files displayed by the browser. The manifest is an index recording each file's origin, role and content hash; original licence text must stay unchanged.


The browser uses the recorded graphics and default sounds; alternative cues
remain in the preview. The [manifest](manifest.json) owns the per-file source,
archive entry, role, size and SHA-256 identity. Original licence text is retained
under licences/. All 13 graphic/audio files are copied byte-for-byte from the
publisher packs; no artwork or sound was created or edited.

## Open and review

From the repository root, run `.\serve-browser.ps1`, then open
http://127.0.0.1:8080/presentation.html. Compare the light/dark interface themes,
seven engine-produced sample games, movement and six sound candidates. The setup
dialog demonstrates layout; configured play is available on the main page.

Sources and included notices are verified as CC0:

- [Board Game Pack](https://kenney.nl/assets/boardgame-pack): the included notice
  identifies Boardgame pack v2 by Kenney Vleugels; provides the round O piece and landing sound.
- [UI Pack](https://kenney.nl/assets/ui-pack): included UI Pack 2.0 notice;
  provides existing scalable X/cell/piece/socket artwork.
- [Interface Sounds](https://kenney.nl/assets/interface-sounds): included 1.0 notice;
  provides placement/result cues and an alternative win cue.
- [Impact Sounds](https://kenney.nl/assets/impact-sounds): included 1.0 notice;
  provides an alternative landing cue.

The publisher page's version summary and the downloaded Board Game Pack notice
differ; the archive hash and original notice identify the actual files used.
System fonts, ordinary HTML/CSS layout and text labels surround existing artwork.
Presentation scales, positions and fades the original images; game state is
unchanged by movement or sound. Numeric labels identify Connect 4 seats beyond colour.

## Reproduce and maintain

Run the downloader in normal PowerShell when candidate archives are missing:

```powershell
.\tools\browser\get-presentation-assets.ps1
python tools/browser/import-presentation-assets.py
python tools/browser/import-presentation-assets.py --check
```

The importer has an explicit list of selected original archive entries and checked
workspace destinations. Keep source ZIPs under ignored .scratch/presentation/incoming.
It retains licence evidence and records imported-file identities; --check compares
the manifest and copied bytes with those actual archives. Browser serving uses only
the selected web/assets files. Additional assets need their own source/licence
inspection and manifest records before inclusion.

Use [browser integration](../../docs/reference/browser/integration.md) for display,
input and validation boundaries.

O correction, 14-10-2026 09:50 (Europe/Paris): the original single09 entry was a
house. The visually inspected single11 entry is the round piece now used for O;
the importer and manifest preserve its original bytes and source identity.

## Love Letter

Nine original geometric SVG card symbols are supplied under the project's MIT
licence. The [origin manifest](games/loveletter/origin.json) records their hashes;
[attribution](games/loveletter/attributions.txt) identifies the source. No TAG,
Flaticon or publisher card images are included. Browser captions retain the
classic rules' ranks independently of the symbols.
