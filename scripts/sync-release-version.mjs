import fs from "node:fs";
import path from "node:path";

const rawVersion = process.argv[2]?.trim();

if (!rawVersion) {
  console.error("Usage: node scripts/sync-release-version.mjs <version>");
  process.exit(1);
}

if (!/^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$/.test(rawVersion)) {
  console.error(`Invalid semver version: ${rawVersion}`);
  process.exit(1);
}

const repoRoot = process.cwd();

function writeJson(filePath, mutator) {
  const absolutePath = path.join(repoRoot, filePath);
  const payload = JSON.parse(fs.readFileSync(absolutePath, "utf8"));
  mutator(payload);
  fs.writeFileSync(absolutePath, `${JSON.stringify(payload, null, 2)}\n`);
}

function replaceVersionInToml(filePath, nextVersion) {
  const absolutePath = path.join(repoRoot, filePath);
  const source = fs.readFileSync(absolutePath, "utf8");
  const updated = source.replace(/^version = ".*"$/m, `version = "${nextVersion}"`);

  if (updated === source) {
    throw new Error(`Failed to update version in ${filePath}`);
  }

  fs.writeFileSync(absolutePath, updated);
}

writeJson("package.json", (payload) => {
  payload.version = rawVersion;
});

writeJson("src-tauri/tauri.conf.json", (payload) => {
  payload.version = rawVersion;
});

replaceVersionInToml("src-tauri/Cargo.toml", rawVersion);

console.log(`Synchronized release version to ${rawVersion}`);
