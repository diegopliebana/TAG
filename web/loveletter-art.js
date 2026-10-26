// Original project symbols in the public distribution; MIT licensed.
export const cardImage = card => new URL(`./assets/games/loveletter/${card || "back"}.svg`, import.meta.url).href;
export const artworkCredit = { label: "Artwork source and MIT licence", path: "./assets/games/loveletter/attributions.txt" };
