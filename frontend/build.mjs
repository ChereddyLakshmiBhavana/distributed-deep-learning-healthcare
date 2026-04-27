import { mkdir, copyFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const sourceDir = process.cwd();
const outputDir = path.resolve('dist');
const apiBaseUrl = process.env.API_BASE_URL || 'http://localhost:5000';

const filesToCopy = ['index.html', 'style.css', 'script.js'];

await mkdir(outputDir, { recursive: true });

for (const fileName of filesToCopy) {
  await copyFile(path.join(sourceDir, fileName), path.join(outputDir, fileName));
}

await writeFile(
  path.join(outputDir, 'config.js'),
  `window.__API_BASE_URL__ = ${JSON.stringify(apiBaseUrl)};\n`,
  'utf8'
);

console.log(`Built frontend to ${outputDir}`);
console.log(`API base URL: ${apiBaseUrl}`);