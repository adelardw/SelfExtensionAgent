import { build } from "esbuild";
const nodeBuiltins = ["assert","child_process","events","fs","fs/promises","http","https",
  "os","path","process","readline","stream","url","util","buffer","crypto","net","tls","zlib",
  "string_decoder","querystring","tty","constants"];
const alias = {};
for (const m of nodeBuiltins) { alias["node:" + m] = "./node_stub.js"; alias[m] = "./node_stub.js"; }
await build({
  entryPoints: ["entry.mjs"], bundle: true, format: "esm", platform: "browser",
  alias, outfile: "../extension/vendor/puppeteer.bundle.js", logLevel: "error", logLimit: 0,
});
console.log("BUILD OK");
