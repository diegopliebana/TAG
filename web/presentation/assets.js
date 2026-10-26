// Resolve the reviewed candidate IDs to locally served, unchanged source files.
export async function loadAssets() {
  const root = new URL("../assets/", import.meta.url);
  const response = await fetch(new URL("manifest.json", root));
  if (!response.ok) throw new Error("The presentation asset inventory could not be loaded.");
  const manifest = await response.json();
  if (manifest.version !== 1) throw new Error("Unsupported asset inventory version.");
  const files = new Map(manifest.assets.map(asset => [asset.id, new URL(asset.path, root).href]));
  const get = id => {
    if (!files.has(id)) throw new Error(`Missing presentation asset: ${id}`);
    return files.get(id);
  };
  const sprites = { x: get("mark-x"), o: get("mark-o"), tttCell: get("ttt-cell"), c4Cell: get("c4-cell"), socket: get("c4-socket"), red: get("c4-red"), yellow: get("c4-yellow") };
  await Promise.all(Object.values(sprites).map(async src => {
    const image = new Image(); image.src = src;
    await image.decode();
  }));
  return { manifest, sprites, get };
}
