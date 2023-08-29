import { readFileSync } from "fs";

export function getTemplate(docstringFormat?: string): string {
  switch (docstringFormat) {
    case "google_summary_only":
      return getTemplateFile("google_summary_only.mustache");
    case "google":
      return getTemplateFile("google.mustache");
    case "sphinx":
      return getTemplateFile("sphinx.mustache");
    case "numpy":
      return getTemplateFile("numpy.mustache");
    default:
      return getTemplateFile("default.mustache");
  }
}

function getTemplateFile(fileName: string): string {
  const filePath = __dirname + "/doc_string/templates/" + fileName;
  return readFileSync(filePath, "utf8");
}
