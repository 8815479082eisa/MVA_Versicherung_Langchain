import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = "C:/Users/mirae/MVA_Versicherung_Langchain_main";
const inputPath = `${root}/artifacts/hallucination-100-current-2026-09-11/final_results.jsonl`;
const outputDir = `${root}/outputs/01a08c78-5de4-7021-907e-5c00b7d06ef2`;
const outputPath = `${outputDir}/hallucination_100_questions_and_generated_answers.xlsx`;

const raw = await fs.readFile(inputPath, "utf8");
const rows = raw.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("Questions and answers");
sheet.showGridLines = false;
sheet.tabColor = "#2563EB";

sheet.getRange("A1:D1").merge();
sheet.getRange("A1").values = [["Hallucination test: 100 generated answers"]];
sheet.getRange("A1").format = {
  font: { name: "Arial", size: 14, bold: true, color: "#172033" },
  verticalAlignment: "center",
};
sheet.getRange("A1:D1").format.rowHeight = 26;

sheet.getRange("A2:D2").merge();
sheet.getRange("A2").values = [[
  "Generated answers are saved drafts immediately before the groundedness stage; all 100 API responses were withheld with HTTP 503 GUARDRAIL_INVALID_OUTPUT."
]];
sheet.getRange("A2").format = {
  font: { name: "Arial", size: 10, italic: true, color: "#5B6472" },
  wrapText: true,
  verticalAlignment: "top",
};
sheet.getRange("A2:D2").format.rowHeight = 34;

const values = [
  ["Query ID", "Question", "Generated answer", "Evaluation result"],
  ...rows.map((row) => [row.query_id, row.question, row.generated_answer, row.result]),
];
sheet.getRange(`A4:D${rows.length + 4}`).values = values;
const table = sheet.tables.add(`A4:D${rows.length + 4}`, true, "GeneratedAnswersTable");
table.style = "TableStyleMedium2";
table.showBandedRows = true;
table.showFilterButton = true;

sheet.getRange(`A4:D${rows.length + 4}`).format.font = { name: "Arial", size: 10, color: "#20242C" };
sheet.getRange("A4:D4").format = {
  fill: "#1F2937",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
sheet.getRange(`A5:A${rows.length + 4}`).format.verticalAlignment = "top";
sheet.getRange(`B5:C${rows.length + 4}`).format = {
  font: { name: "Arial", size: 10, color: "#20242C" },
  verticalAlignment: "top",
  wrapText: true,
};
sheet.getRange(`D5:D${rows.length + 4}`).format = {
  font: { name: "Arial", size: 10, bold: true, color: "#20242C" },
  horizontalAlignment: "center",
  verticalAlignment: "top",
};

sheet.getRange("A:A").format.columnWidth = 18;
sheet.getRange("B:B").format.columnWidth = 62;
sheet.getRange("C:C").format.columnWidth = 100;
sheet.getRange("D:D").format.columnWidth = 18;
sheet.getRange("A4:D4").format.rowHeight = 30;
sheet.getRange(`A5:D${rows.length + 4}`).format.rowHeight = 120;
sheet.freezePanes.freezeRows(4);
sheet.freezePanes.freezeColumns(1);

sheet.getRange(`D5:D${rows.length + 4}`).conditionalFormats.add("containsText", {
  text: "hallucination",
  format: { fill: "#FEE2E2", font: { color: "#991B1B", bold: true } },
});
sheet.getRange(`D5:D${rows.length + 4}`).conditionalFormats.add("containsText", {
  text: "unknown",
  format: { fill: "#FEF3C7", font: { color: "#92400E", bold: true } },
});
sheet.getRange(`D5:D${rows.length + 4}`).conditionalFormats.add("containsText", {
  text: "fallback",
  format: { fill: "#E5E7EB", font: { color: "#374151", bold: true } },
});
sheet.getRange(`D5:D${rows.length + 4}`).conditionalFormats.add("containsText", {
  text: "correct",
  format: { fill: "#DCFCE7", font: { color: "#166534", bold: true } },
});

await fs.mkdir(outputDir, { recursive: true });
const preview = await workbook.render({
  sheetName: "Questions and answers",
  range: "A1:D12",
  scale: 1,
  format: "png",
});
await fs.writeFile(`${outputDir}/hallucination_100_questions_and_generated_answers_preview.png`, new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const inspect = await workbook.inspect({
  kind: "table",
  range: "Questions and answers!A1:D10",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 4,
});
console.log(inspect.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);
console.log(outputPath);
