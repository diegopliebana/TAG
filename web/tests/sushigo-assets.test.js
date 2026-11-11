import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {createHash} from "node:crypto";

test("public TAG Sushi Go! card art retains its recorded byte identities and notice",()=>{
  const root=new URL("../assets/sushigo/",import.meta.url);
  const manifest=JSON.parse(readFileSync(new URL("manifest.json",root),"utf8"));
  assert.equal(manifest.length,13);
  assert.ok(readFileSync(new URL("TAG-LICENSE.txt",root),"utf8").includes("MIT License"));
  for(const item of manifest) {
    assert.match(item.file,/^[A-Za-z0-9_]+\.png$/);
    assert.equal(item.source,`public TAG/data/sushigo/${item.file}`);
    const bytes=readFileSync(new URL(item.file,root));
    assert.equal(bytes.length,item.bytes);
    assert.equal(createHash("sha256").update(bytes).digest("hex"),item.sha256);
  }
});
