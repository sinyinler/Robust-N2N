// Robust-N2N 实验结果汇报 —— 简洁版，按顺序陈列全部结果
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                 // 13.3 x 7.5 in
pres.author = "宋亚栋";
pres.title = "Robust-N2N 实验结果汇报";

const F = "Microsoft YaHei";                 // 中文字体
const NAVY = "1E2761", INK = "222222", MUTED = "6B6B66";
const GREEN = "1D9E75", RED = "C0392B", PINK = "D4537E", BLUE = "378ADD";
const LIGHT = "F4F6F9", LINE = "D8DCE3", HEADbg = "1E2761";
const W = 13.333, H = 7.5;

// —— 通用页眉（无下划线，留白分隔）——
function header(slide, kicker, title) {
  slide.addText(kicker, { x: 0.6, y: 0.34, w: 12, h: 0.32, fontFace: F, fontSize: 12, color: PINK, bold: true, charSpacing: 2, margin: 0 });
  slide.addText(title, { x: 0.6, y: 0.66, w: 12.1, h: 0.7, fontFace: F, fontSize: 27, color: NAVY, bold: true, margin: 0 });
}
// ΔPSNR 单元格（正绿负红）
function dcell(v) {
  const num = (typeof v === "number") ? v : parseFloat(v);
  const txt = (num >= 0 ? "+" : "") + num.toFixed(3);
  return { text: txt, options: { color: num >= 0 ? GREEN : RED, bold: true } };
}
const hcell = t => ({ text: t, options: { fill: { color: HEADbg }, color: "FFFFFF", bold: true, fontFace: F, valign: "middle", align: "center" } });

// ============ Slide 1 · 封面 ============
let s = pres.addSlide();
s.background = { color: NAVY };
s.addText("Robust-N2N", { x: 0.9, y: 2.2, w: 11.5, h: 0.9, fontFace: F, fontSize: 46, color: "FFFFFF", bold: true, margin: 0 });
s.addText("跨视图特征一致性 · 实验结果汇总", { x: 0.9, y: 3.15, w: 11.5, h: 0.6, fontFace: F, fontSize: 24, color: "CADCFC", margin: 0 });
s.addText([
  { text: "N2N 去噪 + 跨视图特征一致性损失　|　5x5 数据 level4 训练　|　50 帧配对评测", options: { breakLine: true, fontSize: 15, color: "AAB6E0" } },
  { text: "内容：权重扫描 → 多 seed 确认 → OOD 泛化测试 → projector 消融 → 诊断与下一步", options: { fontSize: 15, color: "AAB6E0" } },
], { x: 0.9, y: 4.15, w: 11.5, h: 1.0, fontFace: F, margin: 0, lineSpacingMultiple: 1.3 });
s.addText("2026-07-10", { x: 0.9, y: 6.6, w: 4, h: 0.4, fontFace: F, fontSize: 13, color: "8A96C8", margin: 0 });

// ============ Slide 2 · 实验设置与方法 ============
s = pres.addSlide(); s.background = { color: "FFFFFF" };
header(s, "实验设置", "方法与评测协议");
// 左：损失与训练
s.addText([
  { text: "模型 / 损失", options: { bold: true, fontSize: 16, color: NAVY, breakLine: true, paraSpaceAfter: 6 } },
  { text: "轻量 U-Net 去噪器 + 对称 N2N（Charbonnier）", options: { bullet: true, fontSize: 14, color: INK, breakLine: true } },
  { text: "跨视图特征一致性（SimSiam 式：projector + predictor + stop-grad）", options: { bullet: true, fontSize: 14, color: INK, breakLine: true } },
  { text: "L = Charb(f(n1),n2) + Charb(f(n2),n1) + 0.1·Charb(f(n1),f(n2)) + w_feat·L_feat", options: { fontSize: 12.5, color: MUTED, italic: true, breakLine: true, paraSpaceBefore: 4, paraSpaceAfter: 10 } },
  { text: "projector / predictor（训练专用）", options: { bold: true, fontSize: 16, color: NAVY, breakLine: true, paraSpaceAfter: 6 } },
  { text: "g：3 层 1×1 MLP，编码器通道 → 128 维（e3: 64→128，bn: 80→128）", options: { bullet: true, fontSize: 14, color: INK, breakLine: true } },
  { text: "h：128 → 32 → 128（瓶颈 = dim/4，SimSiam 推荐）", options: { bullet: true, fontSize: 14, color: INK, breakLine: true } },
  { text: "二者只属于损失模块，不进 checkpoint；推理只跑 encoder→decoder", options: { bullet: true, fontSize: 14, color: PINK, breakLine: true, paraSpaceAfter: 10 } },
  { text: "训练", options: { bold: true, fontSize: 16, color: NAVY, breakLine: true, paraSpaceAfter: 6 } },
  { text: "5x5 数据集 level4（最干净层），3 epoch，batch 24；同层内配对", options: { bullet: true, fontSize: 14, color: INK } },
], { x: 0.6, y: 1.6, w: 6.4, h: 4.8, fontFace: F, margin: 0, lineSpacingMultiple: 1.12 });
// 右：评测协议卡片
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.3, y: 1.7, w: 5.4, h: 4.6, fill: { color: LIGHT }, line: { color: LINE, width: 1 }, rectRadius: 0.08 });
s.addText([
  { text: "评测协议（降方差）", options: { bold: true, fontSize: 16, color: NAVY, breakLine: true, paraSpaceAfter: 8 } },
  { text: "同场景取 50 帧，逐帧对同一 reference 算 PSNR/MSSIM/r", options: { bullet: true, fontSize: 13.5, color: INK, breakLine: true } },
  { text: "与 N2N 基线在同 50 帧上做逐帧配对 ΔPSNR（抵消场景噪声）", options: { bullet: true, fontSize: 13.5, color: INK, breakLine: true } },
  { text: "判据：ΔPSNR 的 mean > std 才算「稳定赢过 N2N」，否则差异在噪声内", options: { bullet: true, fontSize: 13.5, color: INK, breakLine: true, paraSpaceAfter: 8 } },
  { text: "N2N 基线（level4, 3 epoch）", options: { bold: true, fontSize: 14, color: NAVY, breakLine: true, paraSpaceBefore: 4 } },
  { text: "L = Charb(f(x), 兄弟帧) + 0.01·RTV(f(x))　单向重建", options: { fontSize: 12, color: MUTED, italic: true, breakLine: true } },
  { text: "PSNR 32.509 ± 0.826   MSSIM 0.860   r 0.877", options: { fontSize: 13.5, color: BLUE, bold: true, breakLine: true, paraSpaceAfter: 4 } },
  { text: "注意：Robust 的 w_feat=0 并不等于此基线（它仍有 γ 一致性项、无 RTV、两向重建）", options: { fontSize: 11, color: RED, italic: true } },
], { x: 7.6, y: 1.9, w: 4.8, h: 4.2, fontFace: F, margin: 0, lineSpacingMultiple: 1.18 });

// ============ Slide 3 · 阶段1 权重扫描 ============
s = pres.addSlide(); s.background = { color: "FFFFFF" };
header(s, "阶段 1 · in-distribution（level4, seed42）", "特征损失的「尺度分布」与「强度」扫描");
// 尺度权重双行单元格：名称 + 各尺度比例（顺序 e1/e2/e3/bn）
const wcell = (name, ratio) => ({ text: [
  { text: name, options: { breakLine: true } },
  { text: ratio, options: { fontSize: 8.5, color: MUTED } },
] });
const optsA = { x: 0.6, y: 1.9, w: 6.2, colW: [2.9, 1.75, 1.55], fontFace: F, fontSize: 10.5, border: { pt: 0.5, color: LINE }, align: "center", valign: "middle", rowH: 0.46 };
s.addText("A · 分布对比（w_feat=0.15；e1/e2/e3/bn 归一化权重·和=1）", { x: 0.6, y: 1.55, w: 6.2, h: 0.3, fontFace: F, fontSize: 12, bold: true, color: NAVY, margin: 0 });
s.addTable([
  [hcell("配置（e1/e2/e3/bn 归一化权重）"), hcell("PSNR mean±std"), hcell("ΔPSNR（赢/50）")],
  [wcell("bn 单尺度", "0 / 0 / 0 / 1"), "32.963±0.583", { text: "+0.454（45）", options: { color: GREEN, bold: true } }],
  [wcell("e3 + bn", "0 / 0 / 0.5 / 0.5"), "33.219±0.626", { text: "+0.710（50）", options: { color: GREEN, bold: true } }],
  [wcell("深重浅轻", "0.10 / 0.20 / 0.30 / 0.40"), "32.540±0.988", { text: "+0.031（23）", options: { color: MUTED } }],
  [wcell("均匀", "0.25 / 0.25 / 0.25 / 0.25"), "32.837±0.798", { text: "+0.328（38）", options: { color: MUTED } }],
  [wcell("手调", "0.21 / 0.18 / 0.30 / 0.30"), "32.747±0.807", { text: "+0.238（31）", options: { color: MUTED } }],
], optsA);
const optsB = { x: 6.9, y: 1.9, w: 6.0, colW: [2.3, 1.9, 1.8], fontFace: F, fontSize: 10.5, border: { pt: 0.5, color: LINE }, align: "center", valign: "middle", rowH: 0.36 };
s.addText("B · 强度扫描（固定 e3=bn=0.5 归一化，仅变 w_feat）", { x: 6.9, y: 1.55, w: 6, h: 0.3, fontFace: F, fontSize: 12, bold: true, color: NAVY, margin: 0 });
s.addTable([
  [hcell("w_feat"), hcell("PSNR mean±std"), hcell("ΔPSNR（赢/50）")],
  ["0（特征关）", "31.634±0.778", { text: "−0.874（7）", options: { color: RED, bold: true } }],
  ["0.05", "32.241±0.868", { text: "−0.268（11）", options: { color: RED } }],
  ["0.1", "32.268±0.730", { text: "−0.241（11）", options: { color: RED } }],
  ["0.15", "33.219±0.626", { text: "+0.710（50）", options: { color: GREEN, bold: true } }],
  ["0.2", "32.938±1.052", { text: "+0.429（49）", options: { color: GREEN } }],
  ["0.3", "33.271±0.546", { text: "+0.762（50）", options: { color: GREEN, bold: true } }],
  ["0.5", "33.226±0.803", { text: "+0.717（48）", options: { color: GREEN, bold: true } }],
], optsB);
s.addText("注：w_feat=0 ≠ N2N 基线。它仍是「两向对称重建 + γ=0.1 一致性、无 RTV」；基线是「单向重建 + RTV 0.01」。", { x: 6.9, y: 4.65, w: 6.0, h: 0.55, fontFace: F, fontSize: 10, italic: true, color: RED, margin: 0, lineSpacingMultiple: 1.1 });
s.addText([
  { text: "初步发现（seed42，单 seed，仅供圈定范围）：", options: { bold: true, color: NAVY, breakLine: true } },
  { text: "① 分布上 e3+bn 最好，加浅层(e1/e2)全部落进噪声；② 强度需 w_feat≥0.15，太弱(0/0.05/0.1)反而输给基线；", options: { color: INK, breakLine: true } },
  { text: "③ 同一 base 下 w_feat 0→0.3 提升 +1.6 dB，这是特征损失的净效果；但 w_feat=0 低于 N2N 的 0.87 dB 主要来自 base 差异（γ 项 / 无 RTV / 两向重建），不能全归因于「缺特征损失」。", options: { color: INK } },
], { x: 0.6, y: 6.1, w: 12.1, h: 1.1, fontFace: F, fontSize: 11.5, margin: 0, lineSpacingMultiple: 1.1 });

// ============ Slide 4 · 阶段2 多 seed 确认 ============
s = pres.addSlide(); s.background = { color: "FFFFFF" };
header(s, "阶段 2 · 多 seed 确认", "seed42 的「冠军」其实是运气");
s.addTable([
  [hcell("配置（e3+bn）"), hcell("seed42"), hcell("seed43"), hcell("seed44"), hcell("跨 seed 均值±std")],
  ["w_feat = 0.3", dcell(0.762), dcell(-0.060), dcell(-0.598), { text: "+0.03 ± 0.69", options: { bold: true, color: MUTED } }],
  ["w_feat = 0.15", dcell(0.710), dcell(-2.374), dcell(-0.369), { text: "−0.68 ± 1.57", options: { bold: true, color: RED } }],
], { x: 0.6, y: 1.75, w: 6.3, colW: [1.9, 1.1, 1.1, 1.1, 1.1], fontFace: F, fontSize: 12, border: { pt: 0.5, color: LINE }, align: "center", valign: "middle", rowH: 0.45 });
s.addText("表内为 ΔPSNR（vs N2N），单位 dB；尺度固定 e3=0.5, bn=0.5（归一化，和=1）", { x: 0.6, y: 3.15, w: 6.3, h: 0.5, fontFace: F, fontSize: 10.5, italic: true, color: MUTED, margin: 0, lineSpacingMultiple: 1.1 });
// 柱状图：两配置 × 三 seed
s.addChart(pres.charts.BAR, [
  { name: "seed42", labels: ["w_feat=0.3", "w_feat=0.15"], values: [0.762, 0.710] },
  { name: "seed43", labels: ["w_feat=0.3", "w_feat=0.15"], values: [-0.060, -2.374] },
  { name: "seed44", labels: ["w_feat=0.3", "w_feat=0.15"], values: [-0.598, -0.369] },
], {
  x: 7.1, y: 1.6, w: 5.7, h: 4.4, barDir: "col", chartColors: ["85B7EB", "F0997B", "D4537E"],
  showLegend: true, legendPos: "b", legendColor: INK, legendFontFace: F, legendFontSize: 11,
  showTitle: true, title: "跨 seed ΔPSNR（dB）", titleColor: NAVY, titleFontFace: F, titleFontSize: 13,
  catAxisLabelColor: INK, catAxisLabelFontFace: F, catAxisLabelFontSize: 11,
  valAxisLabelColor: MUTED, valAxisMinVal: -3, valAxisMaxVal: 1, valGridLine: { color: "E8EBF0", size: 0.5 },
  chartArea: { fill: { color: "FFFFFF" } },
});
s.addText([
  { text: "结论：", options: { bold: true, color: NAVY } },
  { text: "w_feat=0.3 跨 3 个 seed = +0.03±0.69 dB —— 统计上就是打平；seed42 的 +0.76 是单 seed 运气。w_feat=0.15 更差且极不稳（seed43 崩 −2.37）。多 seed 成功挡住了假信号。", options: { color: INK } },
], { x: 0.6, y: 5.35, w: 6.3, h: 1.7, fontFace: F, fontSize: 12.5, margin: 0, lineSpacingMultiple: 1.2 });

// ============ Slide 5 · 阶段3 OOD 测试 ============
s = pres.addSlide(); s.background = { color: "FFFFFF" };
header(s, "阶段 3 · OOD 泛化测试（w_feat=0.3, 3 seeds）", "结果与假设完全相反：越 OOD 越差");
s.addTable([
  [hcell("测试层"), hcell("seed42"), hcell("seed43"), hcell("seed44"), hcell("均值")],
  [{ text: "level1（最OOD）", options: { fontFace: F } }, dcell(-0.794), dcell(-1.370), dcell(-2.725), { text: "−1.63", options: { bold: true, color: RED } }],
  [{ text: "level2", options: { fontFace: F } }, dcell(0.534), dcell(-1.030), dcell(-2.981), { text: "−1.16", options: { bold: true, color: RED } }],
  [{ text: "level3", options: { fontFace: F } }, dcell(0.928), dcell(-0.408), dcell(-1.674), { text: "−0.38", options: { bold: true, color: RED } }],
  [{ text: "level4（见过）", options: { fontFace: F } }, dcell(0.762), dcell(-0.060), dcell(-0.598), { text: "+0.03", options: { bold: true, color: MUTED } }],
], { x: 0.6, y: 1.75, w: 6.2, colW: [2.0, 1.05, 1.05, 1.05, 1.05], fontFace: F, fontSize: 11.5, border: { pt: 0.5, color: LINE }, align: "center", valign: "middle", rowH: 0.42 });
s.addText("表内为 ΔPSNR = Robust − N2N（dB）；配置固定 w_feat=0.3, e3=0.5, bn=0.5（归一化）。level1 绝对值：N2N 29.68 vs Robust 26.95~28.88，全线不如。", { x: 0.6, y: 3.75, w: 6.2, h: 0.7, fontFace: F, fontSize: 10.5, italic: true, color: MUTED, margin: 0, lineSpacingMultiple: 1.1 });
// 折线：ΔPSNR vs level（4→1）
s.addChart(pres.charts.LINE, [
  { name: "seed42", labels: ["level4", "level3", "level2", "level1"], values: [0.762, 0.928, 0.534, -0.794] },
  { name: "seed43", labels: ["level4", "level3", "level2", "level1"], values: [-0.060, -0.408, -1.030, -1.370] },
  { name: "seed44", labels: ["level4", "level3", "level2", "level1"], values: [-0.598, -1.674, -2.981, -2.725] },
  { name: "均值", labels: ["level4", "level3", "level2", "level1"], values: [0.03, -0.38, -1.16, -1.63] },
  { name: "打平线(0)", labels: ["level4", "level3", "level2", "level1"], values: [0, 0, 0, 0] },
], {
  x: 7.0, y: 1.6, w: 5.9, h: 4.3, chartColors: ["85B7EB", "F0997B", "D4537E", "1E2761", "AAB0BB"],
  lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 5,
  showLegend: true, legendPos: "b", legendColor: INK, legendFontFace: F, legendFontSize: 10.5,
  showTitle: true, title: "ΔPSNR 随 OOD 加深单调下降", titleColor: NAVY, titleFontFace: F, titleFontSize: 13,
  catAxisLabelColor: INK, catAxisLabelFontFace: F, catAxisLabelFontSize: 11,
  valAxisLabelColor: MUTED, valAxisTitle: "ΔPSNR (dB)", valGridLine: { color: "E8EBF0", size: 0.5 },
  chartArea: { fill: { color: "FFFFFF" } },
});
s.addText([
  { text: "关键：", options: { bold: true, color: RED } },
  { text: "输入越噪、离训练分布越远，Robust 相对 N2N 越差（三 seed 一致、单调）。特征一致性非但没提升 OOD 鲁棒性，反而损害了它。", options: { color: INK } },
], { x: 0.6, y: 4.55, w: 6.2, h: 1.7, fontFace: F, fontSize: 12.5, margin: 0, lineSpacingMultiple: 1.2 });

// ============ Slide 6 · projector 消融 ============
s = pres.addSlide(); s.background = { color: "FFFFFF" };
header(s, "阶段 4 · projector 消融（e3+bn, w_feat=0.3, 3 seeds）", "projector 在 OOD 上挽回约 1 dB");
s.addTable([
  [hcell("测试层"), hcell("无 projector"), hcell("带 projector"), hcell("改善")],
  [{ text: "level1（最OOD）", options: { fontFace: F } }, { text: "−1.63 ± 0.99", options: { color: RED } }, { text: "−0.67 ± 0.46", options: { color: RED, bold: true } }, { text: "+0.96", options: { color: GREEN, bold: true } }],
  [{ text: "level2", options: { fontFace: F } }, { text: "−1.16 ± 1.76", options: { color: RED } }, { text: "−0.09 ± 0.86", options: { color: MUTED, bold: true } }, { text: "+1.07", options: { color: GREEN, bold: true } }],
  [{ text: "level3", options: { fontFace: F } }, { text: "−0.38 ± 1.30", options: { color: RED } }, { text: "+0.05 ± 0.53", options: { color: MUTED, bold: true } }, { text: "+0.43", options: { color: GREEN, bold: true } }],
  [{ text: "level4（见过）", options: { fontFace: F } }, { text: "+0.03 ± 0.69", options: { color: MUTED } }, { text: "+0.14 ± 0.32", options: { color: MUTED, bold: true } }, { text: "+0.11", options: { color: MUTED } }],
], { x: 0.6, y: 1.7, w: 6.3, colW: [1.75, 1.55, 1.55, 1.0], fontFace: F, fontSize: 11, border: { pt: 0.5, color: LINE }, align: "center", valign: "middle", rowH: 0.4 });
s.addText("ΔPSNR = Robust − N2N（dB），跨 3 seed 均值±std。改善随 OOD 加深而增大。", { x: 0.6, y: 3.6, w: 6.3, h: 0.35, fontFace: F, fontSize: 10.5, italic: true, color: MUTED, margin: 0 });
// 特征健康度小卡片
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 4.05, w: 6.3, h: 1.15, fill: { color: LIGHT }, line: { color: LINE, width: 1 }, rectRadius: 0.06 });
s.addText([
  { text: "机制证据 · 特征健康度（std 占健康值 1/√dim 的比例）", options: { bold: true, fontSize: 12, color: NAVY, breakLine: true, paraSpaceAfter: 4 } },
  { text: "无 projector：e3 31~38%、bn 43~54%（严重压缩）　→　带 projector：e3 83~90%、bn 91~92%（恢复健康）", options: { fontSize: 11.5, color: INK } },
], { x: 0.85, y: 4.2, w: 5.8, h: 0.9, fontFace: F, margin: 0, lineSpacingMultiple: 1.15 });
s.addChart(pres.charts.LINE, [
  { name: "无 projector", labels: ["level4", "level3", "level2", "level1"], values: [0.03, -0.38, -1.16, -1.63] },
  { name: "带 projector", labels: ["level4", "level3", "level2", "level1"], values: [0.14, 0.05, -0.09, -0.67] },
  { name: "打平线(0)", labels: ["level4", "level3", "level2", "level1"], values: [0, 0, 0, 0] },
], {
  x: 7.1, y: 1.6, w: 5.75, h: 4.3, chartColors: ["D4537E", "1D9E75", "AAB0BB"],
  lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 6,
  showLegend: true, legendPos: "b", legendColor: INK, legendFontFace: F, legendFontSize: 11,
  showTitle: true, title: "projector 大幅压平 OOD 衰退", titleColor: NAVY, titleFontFace: F, titleFontSize: 13,
  catAxisLabelColor: INK, catAxisLabelFontFace: F, catAxisLabelFontSize: 11,
  valAxisLabelColor: MUTED, valAxisTitle: "ΔPSNR (dB)", valGridLine: { color: "E8EBF0", size: 0.5 },
  chartArea: { fill: { color: "FFFFFF" } },
});
s.addText([
  { text: "结论：", options: { bold: true, color: NAVY } },
  { text: "projector 有大幅、可测量、三 seed 可复现的作用（特征不被压缩、OOD 挽回约 1 dB、方差减半），并非装饰。", options: { color: INK, breakLine: true } },
  { text: "但仍未赢 N2N：", options: { bold: true, color: RED } },
  { text: "level1 为 −0.67±0.46（|mean|>std，是稳定的输）。projector 把「崩盘」变成「小输」，治标未治本。", options: { color: INK } },
], { x: 0.6, y: 5.4, w: 6.3, h: 1.7, fontFace: F, fontSize: 11.5, margin: 0, lineSpacingMultiple: 1.15 });

// ============ Slide 7 · 诊断与下一步 ============
s = pres.addSlide(); s.background = { color: NAVY };
header(s, "诊断与下一步", "");
s.addText("诊断与下一步", { x: 0.6, y: 0.66, w: 12, h: 0.7, fontFace: F, fontSize: 27, color: "FFFFFF", bold: true, margin: 0 });
// 卡片1 诊断
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.7, w: 5.9, h: 4.9, fill: { color: "273377" }, line: { color: "3A47A0", width: 1 }, rectRadius: 0.08 });
s.addText([
  { text: "根因诊断", options: { bold: true, fontSize: 17, color: "FFD166", breakLine: true, paraSpaceAfter: 8 } },
  { text: "只在 level4 单一噪声层训练，且同级配对（n1、n2 噪声等级相同）", options: { bullet: true, fontSize: 13, color: "E8ECFF", breakLine: true } },
  { text: "特征一致性从没见过噪声等级的变化 → 把特征几何过拟合到 level4 的噪声统计", options: { bullet: true, fontSize: 13, color: "E8ECFF", breakLine: true } },
  { text: "无 projector 时特征被压到健康值的 3~5 成，OOD 上崩盘（−1.63 dB）", options: { bullet: true, fontSize: 13, color: "E8ECFF", breakLine: true } },
  { text: "projector 恢复特征健康、挽回约 1 dB —— 但只是缓解，治标未治本", options: { bullet: true, fontSize: 13, color: "FFD166" } },
], { x: 0.9, y: 2.0, w: 5.3, h: 4.4, fontFace: F, margin: 0, lineSpacingMultiple: 1.25 });
// 卡片2 下一步
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.8, y: 1.7, w: 5.9, h: 4.9, fill: { color: "273377" }, line: { color: "3A47A0", width: 1 }, rectRadius: 0.08 });
s.addText([
  { text: "下一步（回到项目原设计）", options: { bold: true, fontSize: 17, color: "6BE3B8", breakLine: true, paraSpaceAfter: 8 } },
  { text: "N2N 与 Robust 都改在 level2/3/4 上训练（多噪声层），并保留 projector", options: { bullet: true, fontSize: 13, color: "E8ECFF", breakLine: true } },
  { text: "留出 level1 当 noise-OOD，重新测配对 ΔPSNR（3 seed）", options: { bullet: true, fontSize: 13, color: "E8ECFF", breakLine: true } },
  { text: "更强版本（可选）：跨级配对（n1 取 level4、n2 取 level2），逼特征对噪声等级不变", options: { bullet: true, fontSize: 13, color: "E8ECFF", breakLine: true } },
  { text: "判据不变：level1 上跨 seed 均值 > std 且为正，才算 OOD 增益成立", options: { bullet: true, fontSize: 13, color: "E8ECFF" } },
], { x: 7.1, y: 2.0, w: 5.3, h: 4.4, fontFace: F, margin: 0, lineSpacingMultiple: 1.25 });
s.addText("诚实结论：一致性正则需要两个条件才可能在 OOD 兑现 —— ① projector 防止特征压缩（已验证，挽回约 1 dB）；② 多噪声层暴露提供「噪声等级不变性」的学习信号（待验证，lv234 正在重跑）。", { x: 0.6, y: 6.7, w: 12.1, h: 0.6, fontFace: F, fontSize: 12, italic: true, color: "CADCFC", margin: 0, lineSpacingMultiple: 1.1 });

pres.writeFile({ fileName: "Robust_N2N_实验结果汇报.pptx" }).then(f => console.log("written:", f));
