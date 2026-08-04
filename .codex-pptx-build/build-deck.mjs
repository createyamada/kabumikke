import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const W = 1280;
const H = 720;
const OUT = "C:/work/kabumikke/株価予測画面_初心者向け説明書.pptx";
const ASSET = "C:/work/kabumikke/.codex-pptx-build";
const FONT = "Noto Sans JP";

const C = {
  ink: "#0F172A",
  sub: "#475569",
  blue: "#2563EB",
  blueSoft: "#E8F0FE",
  purple: "#7C3AED",
  purpleSoft: "#F1EAFE",
  green: "#15803D",
  greenSoft: "#DCFCE7",
  amber: "#B45309",
  amberSoft: "#FEF3C7",
  red: "#B91C1C",
  redSoft: "#FEE2E2",
  line: "#D9E2EF",
  light: "#F8FAFC",
  white: "#FFFFFF",
};

async function bytes(path) {
  const b = await fs.readFile(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

function box(slide, x, y, w, h, fill = C.white, line = C.line, radius = "rounded-xl") {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width: 1 },
    borderRadius: radius,
  });
}

function txt(slide, text, x, y, w, h, size = 22, color = C.ink, bold = false, align = "left") {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: size,
    typeface: FONT,
    color,
    bold,
    alignment: align,
    verticalAlignment: "middle",
    autoFit: "shrinkText",
    insets: { top: 2, right: 4, bottom: 2, left: 4 },
  };
  return shape;
}

function title(slide, text, page, eyebrow = "株価分析ガイド") {
  txt(slide, eyebrow, 56, 24, 330, 28, 15, C.blue, true);
  txt(slide, text, 56, 54, 1150, 68, 34, C.ink, true);
  slide.shapes.add({
    geometry: "rect",
    position: { left: 56, top: 128, width: 1168, height: 2 },
    fill: C.line,
    line: { style: "solid", fill: C.line, width: 0 },
  });
  txt(slide, String(page).padStart(2, "0"), 1175, 670, 48, 24, 13, C.sub, false, "right");
}

function addImage(slide, blob, x, y, w, h, alt, crop) {
  box(slide, x - 4, y - 4, w + 8, h + 8, C.white, C.line);
  return slide.images.add({
    blob,
    contentType: "image/png",
    alt,
    fit: crop ? "cover" : "contain",
    crop,
    geometry: "roundRect",
    borderRadius: "rounded-lg",
    position: { left: x, top: y, width: w, height: h },
  });
}

function pill(slide, label, x, y, w, fill, color) {
  box(slide, x, y, w, 34, fill, fill, "rounded-full");
  txt(slide, label, x + 6, y + 2, w - 12, 30, 16, color, true, "center");
}

function notes(slide, sources) {
  slide.speakerNotes.textFrame.setText(`[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}`);
  slide.speakerNotes.setVisible(true);
}

async function main() {
  const forecast = await bytes(`${ASSET}/screen-forecast.png`);
  const model = await bytes(`${ASSET}/screen-model.png`);
  const tda = await bytes(`${ASSET}/screen-tda.png`);

  const p = Presentation.create({ slideSize: { width: W, height: H } });

  // 1 — title
  {
    const s = p.slides.add();
    s.background.fill = C.light;
    txt(s, "株価予測画面の\n読み方", 60, 150, 500, 170, 58, C.ink, true);
    txt(s, "回帰分析・バックテスト・TDAを\n初心者でも判断に使える形で理解する", 64, 342, 500, 92, 24, C.sub);
    pill(s, "画面例：銘柄コード 5802", 64, 474, 270, C.blueSoft, C.blue);
    addImage(s, forecast, 620, 72, 600, 570, "日本株の翌営業日予測画面", { left: 0.22, top: 0.19, right: 0, bottom: 0.08 });
    txt(s, "予測値は答えではなく、検証結果と一緒に読む", 64, 610, 500, 34, 18, C.blue, true);
    notes(s, [
      "User-provided screenshot: screen-forecast.png",
      "https://github.com/createyamada/kabunseki_web/blob/master/src/components/Main/Analysis.tsx",
    ]);
  }

  // 2 — overview
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    title(s, "結論：予測株価だけで売買を決めない", 2);
    addImage(s, forecast, 56, 164, 690, 460, "予測画面全体", undefined);
    const steps = [
      ["1", "予測", "予測収益率と価格レンジを見る"],
      ["2", "精度", "改善率・方向一致率を見る"],
      ["3", "安定性", "直近でも誤差が崩れていないか"],
      ["4", "収益性", "買い持ちより優れているか"],
      ["5", "相場構造", "TDAで複雑さを確認する"],
    ];
    steps.forEach(([n, h, b], i) => {
      const y = 162 + i * 92;
      box(s, 790, y, 420, 76, i === 0 ? C.blueSoft : C.light, i === 0 ? C.blue : C.line);
      box(s, 808, y + 15, 44, 44, i === 0 ? C.blue : C.ink, i === 0 ? C.blue : C.ink, "rounded-full");
      txt(s, n, 808, y + 16, 44, 42, 20, C.white, true, "center");
      txt(s, h, 870, y + 8, 120, 28, 19, C.ink, true);
      txt(s, b, 870, y + 34, 315, 30, 16, C.sub);
    });
    txt(s, "5つがそろって初めて「比較的信頼できる候補」になる", 790, 635, 420, 34, 18, C.blue, true);
    notes(s, [
      "User-provided screenshot: screen-forecast.png",
      "https://github.com/createyamada/kabunseki_web/blob/master/src/components/Main/Analysis.tsx",
    ]);
  }

  // 3 — forecast
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    title(s, "最初に見るのは「予測幅」と「予測リターン」", 3);
    addImage(s, forecast, 56, 168, 730, 360, "予測サマリー部分", { left: 0.02, top: 0.31, right: 0.02, bottom: 0.27 });
    pill(s, "この例", 56, 550, 90, C.blueSoft, C.blue);
    txt(s, "終値 ¥2,161.5 → 予測 ¥2,194.69（+1.54%）", 160, 548, 620, 38, 22, C.ink, true);
    txt(s, "80%予測レンジ：¥2,087.83 ～ ¥2,308.47", 160, 590, 620, 34, 19, C.sub);
    txt(s, "どう読む？", 830, 172, 330, 42, 25, C.ink, true);
    const bullets = [
      "+1.54%は上昇予測。ただし精度ではない",
      "レンジ内に下落側も含まれている",
      "「80%」は的中確率80%ではない",
      "採用モデル名より、次ページの評価が重要",
    ];
    bullets.forEach((b, i) => {
      box(s, 830, 232 + i * 78, 380, 62, i === 0 ? C.blueSoft : C.light, C.line);
      txt(s, `${i + 1}`, 844, 245 + i * 78, 28, 34, 18, i === 0 ? C.blue : C.sub, true, "center");
      txt(s, b, 882, 238 + i * 78, 310, 46, 17, C.ink, i === 0);
    });
    box(s, 830, 555, 380, 78, C.amberSoft, C.amber);
    txt(s, "予測レンジが広いほど、価格予測の不確実性は高い", 850, 570, 340, 48, 18, C.amber, true);
    notes(s, ["User-provided screenshot: screen-forecast.png"]);
  }

  // 4 — metrics
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    title(s, "回帰モデルは4つの指標で信頼度を測る", 4);
    addImage(s, forecast, 56, 166, 720, 250, "モデル評価部分", { left: 0.02, top: 0.72, right: 0.02, bottom: 0 });
    const rows = [
      ["価格 RMSE", "¥96.67", "低いほど良い", "大きな外れを重く評価"],
      ["価格 MAE", "¥67.26", "低いほど良い", "平均的な誤差額"],
      ["改善率", "0.91%", "高いほど良い", "5%以上を一つの目安"],
      ["方向一致率", "57.14%", "高いほど良い", "55%以上＋十分な件数"],
    ];
    rows.forEach((r, i) => {
      const y = 446 + i * 52;
      txt(s, r[0], 72, y, 165, 38, 18, C.ink, true);
      txt(s, r[1], 245, y, 120, 38, 19, i >= 2 ? C.blue : C.ink, true);
      pill(s, r[2], 385, y + 2, 145, i >= 2 ? C.greenSoft : C.blueSoft, i >= 2 ? C.green : C.blue);
      txt(s, r[3], 550, y, 230, 38, 16, C.sub);
    });
    txt(s, "この例の読み取り", 835, 172, 330, 40, 25, C.ink, true);
    box(s, 835, 230, 375, 104, C.amberSoft, C.amber);
    txt(s, "方向一致率 57.14%", 855, 242, 330, 34, 24, C.green, true);
    txt(s, "方向予測には一定の優位性", 855, 279, 330, 32, 17, C.sub);
    box(s, 835, 352, 375, 104, C.redSoft, C.red);
    txt(s, "改善率 0.91%", 855, 364, 330, 34, 24, C.red, true);
    txt(s, "単純予測との差は小さい", 855, 401, 330, 32, 17, C.sub);
    box(s, 835, 478, 375, 142, C.light, C.line);
    txt(s, "結論", 855, 490, 100, 30, 18, C.ink, true);
    txt(s, "「方向はやや当たるが、誤差改善は弱い」\n予測株価だけで強気判断はしない", 855, 526, 330, 78, 19, C.ink, true);
    notes(s, ["User-provided screenshot: screen-forecast.png"]);
  }

  // 5 — model comparison
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    title(s, "採用モデルでも、直近で崩れていないか確認する", 5);
    addImage(s, model, 56, 172, 740, 240, "モデル比較表", { left: 0.04, top: 0.02, right: 0.03, bottom: 0.69 });
    txt(s, "見る順番", 842, 174, 320, 36, 24, C.ink, true);
    txt(s, "① ウォークフォワードRMSE：過去を順に検証\n② ホールドアウトRMSE：最後の未使用期間で検証\n③ 2つが低く、近いほど安定", 842, 224, 350, 142, 18, C.sub);
    box(s, 56, 450, 520, 132, C.blueSoft, C.blue);
    txt(s, "乖離率 ＝ ホールドアウト ÷ ウォークフォワード", 78, 465, 480, 34, 19, C.ink, true);
    txt(s, "Ridge回帰：4.17% ÷ 1.86% ＝ 2.24倍", 78, 508, 480, 34, 24, C.red, true);
    txt(s, "目安：1.25倍以内なら比較的安定", 78, 548, 480, 26, 16, C.sub);
    box(s, 610, 450, 600, 132, C.redSoft, C.red);
    txt(s, "この例は直近で誤差が約2.24倍", 634, 466, 550, 36, 25, C.red, true);
    txt(s, "採用モデルでも、最近の相場では精度が落ちている可能性。\n「採用」表示だけでは信頼できない。", 634, 512, 550, 56, 18, C.ink);
    txt(s, "低いほど良い　＋　2つの差が小さいほど良い", 56, 624, 760, 32, 19, C.blue, true);
    notes(s, ["User-provided screenshot: screen-model.png"]);
  }

  // 6 — backtest
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    title(s, "利益が出ても、買い持ちに負ければ優位性は弱い", 6);
    addImage(s, model, 56, 170, 700, 420, "バックテストと株価推移", { left: 0.03, top: 0.27, right: 0.03, bottom: 0.04 });
    const values = [
      ["戦略リターン", "111.86%", C.green],
      ["買い持ち", "163.19%", C.blue],
      ["シャープレシオ", "1.74", C.green],
      ["最大ドローダウン", "-34.19%", C.red],
    ];
    values.forEach((v, i) => {
      const y = 174 + i * 92;
      txt(s, v[0], 808, y, 230, 26, 17, C.sub, true);
      txt(s, v[1], 1030, y - 4, 170, 36, 26, v[2], true, "right");
      s.shapes.add({ geometry: "rect", position: { left: 808, top: y + 42, width: 392, height: 1 }, fill: C.line, line: { style: "solid", fill: C.line, width: 0 } });
    });
    box(s, 808, 556, 392, 82, C.amberSoft, C.amber);
    txt(s, "収益はプラスでも買い持ちに約51ポイント負け。\n最大下落も大きいため、優位性は弱い。", 826, 568, 356, 58, 17, C.amber, true);
    notes(s, ["User-provided screenshot: screen-model.png"]);
  }

  // 7 — TDA
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    title(s, "TDAは方向ではなく、相場の複雑さを測る", 7, "回帰分析を補助するTDA");
    addImage(s, tda, 56, 170, 700, 430, "TDA分析画面", undefined);
    txt(s, "この例", 810, 172, 160, 34, 22, C.ink, true);
    pill(s, "中程度の複雑度", 810, 218, 230, C.purpleSoft, C.purple);
    txt(s, "ループ強度 7.86%", 810, 270, 360, 38, 26, C.purple, true);
    txt(s, "バックエンドの区分", 810, 328, 320, 30, 18, C.ink, true);
    const bands = [
      ["5%未満", "低い", C.greenSoft, C.green],
      ["5～15%", "中程度", C.amberSoft, C.amber],
      ["15%以上", "高い", C.redSoft, C.red],
    ];
    bands.forEach((b, i) => {
      box(s, 810, 370 + i * 58, 360, 44, b[2], b[2]);
      txt(s, b[0], 826, 375 + i * 58, 120, 32, 17, b[3], true);
      txt(s, b[1], 1010, 375 + i * 58, 130, 32, 17, b[3], true, "right");
    });
    box(s, 810, 558, 360, 76, C.blueSoft, C.blue);
    txt(s, "高複雑度なら予測を割り引く。\nただし低複雑度＝上昇ではない。", 828, 570, 325, 50, 17, C.blue, true);
    notes(s, [
      "User-provided screenshot: screen-tda.png",
      "Backend heuristic: loop_strength <5% low, <15% moderate, otherwise high",
    ]);
  }

  // 8 — case assessment
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    title(s, "5802の判定：方向予測は良いが、総合信頼度は低い", 8, "画面の数値を実際に評価");
    const rows = [
      ["改善率", "0.91%", "5%以上", "未達", C.redSoft, C.red],
      ["方向一致率", "57.14%", "55%以上", "合格", C.greenSoft, C.green],
      ["RMSE乖離", "2.24倍", "1.25倍以内", "未達", C.redSoft, C.red],
      ["シャープレシオ", "1.74", "1.0以上", "合格", C.greenSoft, C.green],
      ["戦略 vs 買い持ち", "111.86% < 163.19%", "戦略が上", "未達", C.redSoft, C.red],
      ["最大ドローダウン", "-34.19%", "許容範囲内", "高リスク", C.redSoft, C.red],
      ["TDA", "中程度", "高複雑度でない", "合格", C.greenSoft, C.green],
    ];
    txt(s, "指標", 70, 160, 260, 34, 17, C.sub, true);
    txt(s, "画面の値", 360, 160, 270, 34, 17, C.sub, true);
    txt(s, "目安", 680, 160, 230, 34, 17, C.sub, true);
    txt(s, "判定", 1030, 160, 130, 34, 17, C.sub, true, "center");
    rows.forEach((r, i) => {
      const y = 202 + i * 57;
      s.shapes.add({ geometry: "rect", position: { left: 62, top: y + 49, width: 1140, height: 1 }, fill: C.line, line: { style: "solid", fill: C.line, width: 0 } });
      txt(s, r[0], 70, y, 260, 44, 18, C.ink, true);
      txt(s, r[1], 360, y, 280, 44, 18, C.ink, r[3] !== "合格");
      txt(s, r[2], 680, y, 250, 44, 17, C.sub);
      pill(s, r[3], 1030, y + 5, 130, r[4], r[5]);
    });
    box(s, 62, 618, 1140, 52, C.amberSoft, C.amber);
    txt(s, "判断：+1.54%の上昇予測は出ているが、モデル単独では「強い買い根拠」にしない。監視候補として扱う。", 82, 626, 1100, 36, 20, C.amber, true);
    notes(s, [
      "User-provided screenshots: screen-forecast.png, screen-model.png, screen-tda.png",
      "Thresholds are instructional heuristics and require strategy-specific validation.",
    ]);
  }

  // 9 — flow
  {
    const s = p.slides.add();
    s.background.fill = C.white;
    title(s, "初心者は、この順番で判断すれば迷いにくい", 9, "6ステップの確認手順");
    const items = [
      ["1", "予測方向", "プラス／マイナス"],
      ["2", "改善率", "5%以上か"],
      ["3", "方向一致", "55%以上か"],
      ["4", "直近安定性", "RMSE乖離1.25倍以内"],
      ["5", "運用成績", "買い持ち超え\nSharpe 1以上"],
      ["6", "TDA", "高複雑度ではないか"],
    ];
    for (let i = 0; i < items.length - 1; i++) {
      s.shapes.add({
        geometry: "rightArrow",
        position: { left: 176 + i * 198, top: 330, width: 55, height: 34 },
        fill: C.line,
        line: { style: "solid", fill: C.line, width: 0 },
      });
    }
    items.forEach((it, i) => {
      const x = 46 + i * 198;
      box(s, x, 220, 170, 250, i === 0 ? C.blueSoft : C.light, i === 0 ? C.blue : C.line);
      box(s, x + 55, 242, 60, 60, i === 0 ? C.blue : C.ink, i === 0 ? C.blue : C.ink, "rounded-full");
      txt(s, it[0], x + 55, 245, 60, 54, 26, C.white, true, "center");
      txt(s, it[1], x + 14, 320, 142, 36, 19, C.ink, true, "center");
      txt(s, it[2], x + 14, 366, 142, 70, 16, C.sub, false, "center");
    });
    box(s, 172, 535, 936, 84, C.blueSoft, C.blue);
    txt(s, "途中で重大な未達があれば、投資額を減らす・見送る・他の情報を確認する", 198, 550, 884, 54, 22, C.blue, true, "center");
    notes(s, ["Decision sequence derived from the metrics displayed in Analysis.tsx"]);
  }

  // 10 — final checklist
  {
    const s = p.slides.add();
    s.background.fill = C.light;
    title(s, "最終チェック：数字が同じ方向を示しているか", 10, "投資判断のまとめ");
    const cols = [
      { x: 56, fill: C.greenSoft, line: C.green, head: "比較的信頼しやすい", items: ["改善率 5%以上", "方向一致率 55%以上", "直近RMSEが安定", "戦略が買い持ち超え", "Sharpe 1以上", "TDAが高複雑度でない"] },
      { x: 448, fill: C.amberSoft, line: C.amber, head: "慎重に扱う", items: ["改善率 0～5%", "方向一致率 50～55%", "予測レンジが広い", "RMSEが直近で悪化", "大きなドローダウン"] },
      { x: 840, fill: C.redSoft, line: C.red, head: "信頼しにくい", items: ["改善率がマイナス", "方向一致率 50%未満", "戦略リターンがマイナス", "買い持ちに大幅負け", "Sharpe 0以下"] },
    ];
    cols.forEach((c) => {
      box(s, c.x, 172, 352, 382, c.fill, c.line);
      txt(s, c.head, c.x + 22, 194, 308, 40, 24, c.line, true, "center");
      c.items.forEach((it, i) => {
        txt(s, `✓`, c.x + 28, 258 + i * 48, 28, 32, 18, c.line, true, "center");
        txt(s, it, c.x + 62, 254 + i * 48, 260, 40, 17, C.ink);
      });
    });
    txt(s, "予測は「候補を絞る道具」。決算・材料・市場環境・損失許容額も確認して、最終判断は自分で行う。", 90, 592, 1100, 54, 22, C.ink, true, "center");
    txt(s, "過去データの成績は、将来の利益を保証しません。", 90, 650, 1100, 28, 16, C.red, true, "center");
    notes(s, ["Thresholds are instructional heuristics and must be validated for each investment strategy."]);
  }

  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(OUT);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
