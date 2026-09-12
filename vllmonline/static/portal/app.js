/* vLLMonline 演示门户 · 前端逻辑
 * 轮询后端 API → 渲染卡片/图表。无框架，无构建。
 */

const API = {
    models: "/api/models",
    canaries: "/api/canary",
    evals: "/api/eval",
    healthz: "/healthz",
    readyz: "/readyz",
    metrics: "/metrics",
    chat: "/v1/chat/completions",
};

const POLL_MS = 2000;
const MAX_POINTS = 120; // 折线图最多保留的点数（2s × 120 = 4 分钟）

const COLORS = { v1: "#3498db", v2: "#e67e22" };

/* MI300X 实测快照（Qwen2.5-7B vs Qwen2.5-1.5B，AMD MI300X 192GB 单卡）
 * 数据库为空时回填展示；字段结构与 GET /api/eval、GET /api/canary 响应一致。
 * 数据来源：scripts/run_ab_eval.py 真实运行结果（BLOG.md §七 / README 实测数据）。
 */
const EVAL_SNAPSHOT = {
    id: "mi300x-ab-n49",
    sample_count: 49,
    score_v1_mean: 0.9207,
    score_v2_mean: 0.8578,
    p_value: 0.0001,
    significant: true,
    effect_size: -0.8315,
    recommendation: "rollback",
    dimension_scores: {
        v1: { accuracy: 0.8765, completeness: 0.8857, safety: 1.0 },
        v2: { accuracy: 0.7816, completeness: 0.7918, safety: 1.0 },
    },
    created_at: "2026-09-11",
};

const CANARY_SNAPSHOT = {
    id: "canary-9543394eec75",
    model_v1_id: "qwen-7b-v1",
    model_v2_id: "qwen-7b-v2",
    strategy: "gray",
    stages: [0.1, 0.3, 1.0],
    current_stage_index: 1,
    traffic_split: { "qwen-7b-v1": 0.7, "qwen-7b-v2": 0.3 },
    status: "ROLLED_BACK",
    started_at: "2026-09-11",
};

// ── 全局状态 ─────────────────────────────
let splitChart = null;      // 流量配比环形图
let requestsChart = null;   // 请求数折线图
let dimsChart = null;       // 三维度条形图
let activeDeploy = null;    // 当前进行中的部署
let series = { t: [], v1: [], v2: [] };  // 折线图历史
let lastCounts = null;      // 上次指标快照（算增量）

// ── 工具 ─────────────────────────────────
const $ = (sel) => document.querySelector(sel);

function toast(msg, type = "info") {
    const el = document.createElement("div");
    el.className = `toast ${type === "ok" ? "ok" : type === "err" ? "err" : ""}`;
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 3500);
}

async function fetchJSON(url, opts) {
    const r = await fetch(url, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
    return body;
}

function fmtTime(iso) {
    if (!iso) return "-";
    return iso.replace("T", " ").slice(5, 19);
}

// ── Hero 状态徽章 ────────────────────────
async function refreshStatus() {
    const set = (id, ok) => {
        const el = $(id);
        el.className = `badge ${ok ? "online" : "offline"}`;
    };
    try {
        const h = await fetch(API.healthz);
        set("#badge-platform", h.ok);
    } catch { set("#badge-platform", false); }

    try {
        const r = await fetch(API.readyz);
        const body = await r.json();
        // readyz 的 checks.vllm 只探默认 backend（v1）
        set("#badge-v1", body.checks?.vllm === "ok");
        set("#badge-v2", body.checks?.vllm === "ok"); // 简化：v2 与 v1 同探针
    } catch {
        set("#badge-v1", false); set("#badge-v2", false);
    }
}

// ── 模型面板 ─────────────────────────────
async function refreshModels() {
    const data = await fetchJSON(API.models);
    const grid = $("#model-grid");

    if (!data.models?.length) {
        grid.innerHTML = `<div class="empty-hint">暂无模型。<br>用 API 注册：<code>POST /api/models/register</code>（见 README）</div>`;
        return;
    }

    grid.innerHTML = data.models.map((m) => {
        const st = m.state;
        const actions = buildActions(m);
        return `
        <div class="model-card">
            <div class="model-card-top">
                <span class="model-name">${m.id}</span>
                <span class="state-badge state-${st}">${st}</span>
            </div>
            <div class="model-meta">
                参数量 <b>${m.params_billion}B</b> · ${m.dtype}${m.quantization ? " + " + m.quantization : ""}
                <br>权重 <b>${(m.weight_gb ?? 0).toFixed(1)}GB</b> · KV cache <b>${(m.kv_cache_budget_gb ?? 0).toFixed(2)}GB</b> · GPU ${m.gpu_id ?? "-"}
                <br><span class="mono" style="font-size:11px">${m.endpoint}</span>
            </div>
            <div class="model-actions">${actions}</div>
        </div>`;
    }).join("");
}

function buildActions(m) {
    const st = m.state;
    const btn = (action, label, enabled, cls) =>
        `<button class="btn ${cls || ""}" ${enabled ? "" : "disabled"}
            onclick="modelAction('${m.id}','${action}')">${label}</button>`;
    return (
        btn("load", "Load", st === "IDLE", "btn-green") +
        btn("sleep", "Sleep", st === "ACTIVE") +
        btn("wake", "Wake", st === "SLEEPING") +
        btn("unload", "Unload", st === "ACTIVE" || st === "SLEEPING", "btn-red")
    );
}

// 挂到 window 供 inline onclick 调用
window.modelAction = async function (id, action) {
    try {
        await fetchJSON(`${API.models}/${id}/${action}`, { method: "POST" });
        toast(`${id} → ${action} 成功`, "ok");
    } catch (e) {
        toast(`${id} → ${action} 失败：${e.message}`, "err");
    }
    refreshModels();
};

// ── 灰度控制台 ───────────────────────────
async function refreshCanary() {
    const data = await fetchJSON(API.canaries);
    const tbody = $("#table-canary tbody");

    // 数据库为空时回填实测快照（仅进历史表，不参与按钮/活跃部署逻辑）
    let deployments = data.deployments || [];
    $("#canary-source").hidden = deployments.length > 0;
    if (!deployments.length) deployments = [CANARY_SNAPSHOT];

    // 找进行中的部署
    activeDeploy = deployments.find((d) => d.status === "IN_PROGRESS") || null;
    updateCanaryButtons();

    // 部署历史表
    tbody.innerHTML = deployments.map((d) => {
        const split = Object.entries(d.traffic_split || {})
            .map(([k, v]) => `${k.split("-").pop()}:${(v * 100).toFixed(0)}%`).join(" / ");
        return `<tr>
            <td class="mono">${d.id.slice(0, 18)}…</td>
            <td class="mono">${d.model_v1_id || "-"}</td>
            <td class="mono">${d.model_v2_id || "-"}</td>
            <td><span class="state-badge state-${d.status === "IN_PROGRESS" ? "ACTIVE" : d.status === "ROLLED_BACK" ? "SLEEPING" : "IDLE"}">${d.status}</span></td>
            <td class="mono">${split || "-"}</td>
            <td>${fmtTime(d.started_at)}</td>
        </tr>`;
    }).join("") || `<tr><td colspan="6" style="text-align:center;color:var(--text-dim)">暂无部署记录</td></tr>`;

    if (activeDeploy) {
        renderActiveDeploy(activeDeploy);
    } else {
        $("#deploy-info").innerHTML =
            `<div class="empty-hint" style="border:none;padding:6px 0">暂无进行中的灰度部署<br>点击上方「启动灰度」开始演示</div>`;
        // 无部署时饼图显示 100:0
        renderSplitChart({ v1: 1.0, v2: 0.0 });
        highlightStage(null);
    }
}

function updateCanaryButtons() {
    const has = !!activeDeploy;
    $("#btn-start-canary").disabled = has;
    $("#btn-advance").disabled = !has;
    $("#btn-rollback").disabled = !has;
}

function renderActiveDeploy(d) {
    // 饼图：traffic_split {model_id: weight}
    const entries = Object.entries(d.traffic_split || {});
    let v1w = 0, v2w = 0;
    for (const [id, w] of entries) {
        if (id === d.model_v1_id) v1w = w;
        if (id === d.model_v2_id) v2w = w;
    }
    renderSplitChart({ v1: v1w, v2: v2w });

    // 阶段条
    const idx = d.current_stage_index ?? 0;
    const stages = d.stages || [0.1, 0.3, 1.0];
    const stageNames = ["INIT", "STAGE_10%", "STAGE_30%", "STAGE_100%", "COMPLETED"];
    // index i 对应 stageNames[i+1]（跳过 INIT）
    const currentName = idx >= stages.length ? "COMPLETED" : stageNames[Math.min(idx + 1, 4)];
    highlightStage(currentName);

    // 信息卡
    $("#deploy-info").innerHTML = `
        <div class="kv"><span class="k">部署 ID</span><span class="mono">${d.id}</span></div>
        <div class="kv"><span class="k">v1（baseline）</span><span class="mono">${d.model_v1_id}</span></div>
        <div class="kv"><span class="k">v2（候选）</span><span class="mono">${d.model_v2_id}</span></div>
        <div class="kv"><span class="k">当前阶段</span><span>${currentName}</span></div>
        <div class="kv"><span class="k">流量配比</span><span class="mono">v1 ${(v1w * 100).toFixed(0)}% : v2 ${(v2w * 100).toFixed(0)}%</span></div>
        <div class="kv"><span class="k">阶段序列</span><span class="mono">${stages.join(" → ")}</span></div>`;
}

function highlightStage(current) {
    const names = ["INIT", "STAGE_10%", "STAGE_30%", "STAGE_100%", "COMPLETED"];
    const curIdx = current ? names.indexOf(current) : -1;
    document.querySelectorAll("#stage-bar .stage").forEach((el, i) => {
        el.className = "stage" + (i < curIdx ? " done" : i === curIdx ? " current" : "");
    });
}

function renderSplitChart(split) {
    const total = split.v1 + split.v2;
    const data = total > 0 ? [split.v1, split.v2] : [1, 0];
    if (!splitChart) {
        splitChart = new Chart($("#chart-split"), {
            type: "doughnut",
            data: {
                labels: ["v1 (baseline)", "v2 (candidate)"],
                datasets: [{
                    data,
                    backgroundColor: [COLORS.v1, COLORS.v2],
                    borderColor: "#1a2332",
                    borderWidth: 3,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: "62%",
                plugins: {
                    legend: { position: "bottom", labels: { color: "#8b9bb4", padding: 14 } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => ` ${ctx.label}: ${(ctx.parsed * 100).toFixed(1)}%`,
                        },
                    },
                },
            },
        });
    } else {
        splitChart.data.datasets[0].data = data;
        splitChart.update("none");
    }
}

// 灰度操作按钮
$("#btn-start-canary").addEventListener("click", async () => {
    try {
        const models = await fetchJSON(API.models);
        const byName = {};
        for (const m of models.models || []) (byName[m.model_name] = byName[m.model_name] || []).push(m);
        // 取第一组有两个版本的 model_name
        let v1 = null, v2 = null;
        for (const [name, list] of Object.entries(byName)) {
            if (list.length >= 2) {
                const active = list.filter((m) => m.state === "ACTIVE");
                if (active.length >= 2) { v1 = active[0]; v2 = active[1]; break; }
            }
        }
        if (!v1 || !v2) throw new Error("需要两个 ACTIVE 的同组模型（如 qwen-7b-v1 + qwen-7b-v2）。请先在模型面板 Load 两个版本。");

        await fetchJSON(`${API.canaries}/start`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model_v1: v1.id, model_v2: v2.id,
                strategy: "gradual", stages: [0.1, 0.3, 1.0],
            }),
        });
        toast(`灰度已启动：${v1.id} ↔ ${v2.id}，v2 占 10%`, "ok");
    } catch (e) { toast(`启动灰度失败：${e.message}`, "err"); }
    refreshCanary();
});

$("#btn-advance").addEventListener("click", async () => {
    if (!activeDeploy) return;
    try {
        const r = await fetchJSON(`${API.canaries}/${activeDeploy.id}/advance`, { method: "POST" });
        toast(r.advanced ? `已推进到 ${r.current_stage}` : "已完成全部阶段", "ok");
    } catch (e) { toast(`推进失败：${e.message}`, "err"); }
    refreshCanary();
});

$("#btn-rollback").addEventListener("click", async () => {
    if (!activeDeploy) return;
    if (!confirm("确认回滚？v2 将被 drain + sleep，流量 100% 切回 v1。")) return;
    const t0 = performance.now();
    try {
        await fetchJSON(
            `${API.canaries}/${activeDeploy.id}/rollback?reason=${encodeURIComponent("portal_demo_rollback")}`,
            { method: "POST" }
        );
        const ms = (performance.now() - t0).toFixed(0);
        toast(`回滚完成，耗时 ${ms}ms —— 流量已 100% 回到 v1`, "ok");
    } catch (e) { toast(`回滚失败：${e.message}`, "err"); }
    refreshCanary();
    refreshModels();
});

// 发测试请求（演示流量）
$("#btn-send-traffic").addEventListener("click", async () => {
    const btn = $("#btn-send-traffic");
    btn.disabled = true;
    btn.textContent = "⏳ 发送中…";
    let ok = 0, fail = 0;
    const prompts = Array.from({ length: 20 }, (_, i) => `demo request ${i}: ${["用一句话介绍自己", "1+1等于几", "写一个词：秋天", "什么是 GPU"][i % 4]}`);
    await Promise.all(prompts.map(async (p) => {
        try {
            const r = await fetch(API.chat, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model: "qwen-7b", messages: [{ role: "user", content: p }], max_tokens: 8, temperature: 0.7 }),
            });
            (r.ok ? ok++ : fail++);
        } catch { fail++; }
    }));
    toast(`20 个请求完成：${ok} 成功 / ${fail} 失败。观察下方监控曲线的 v1/v2 分布 →`, fail ? "err" : "ok");
    btn.disabled = false;
    btn.textContent = "🔥 发送 20 个测试请求";
});

// ── 监控面板（解析 /metrics 文本）────────
async function refreshMetrics() {
    const text = await fetch(API.metrics).then((r) => r.text());

    const grab = (version, status) => {
        const re = new RegExp(
            `vllmonline_requests_total\\{[^}]*model_version="${version}"[^}]*status="${status}"[^}]*\\}\\s+([\\d.e+]+)`
        );
        const m = text.match(re);
        return m ? parseFloat(m[1]) : 0;
    };

    const v1ok = grab("v1", "success");
    const v2ok = grab("v2", "success");
    const v2err = grab("v2", "error");

    $("#stat-v1-count").textContent = v1ok.toFixed(0);
    $("#stat-v2-count").textContent = v2ok.toFixed(0);
    $("#stat-v2-errors").textContent = v2err.toFixed(0);

    // 折线图（相对页面打开时的累计差值）
    if (!lastCounts) lastCounts = { v1: v1ok, v2: v2ok };
    const now = new Date();
    series.t.push(now.toTimeString().slice(0, 8));
    series.v1.push(v1ok - lastCounts.v1);
    series.v2.push(v2ok - lastCounts.v2);
    if (series.t.length > MAX_POINTS) { series.t.shift(); series.v1.shift(); series.v2.shift(); }
    renderRequestsChart();
}

function renderRequestsChart() {
    if (!requestsChart) {
        requestsChart = new Chart($("#chart-requests"), {
            type: "line",
            data: {
                labels: series.t,
                datasets: [
                    { label: "v1 (baseline)", data: series.v1, borderColor: COLORS.v1, backgroundColor: "rgba(52,152,219,0.08)", fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 },
                    { label: "v2 (candidate)", data: series.v2, borderColor: COLORS.v2, backgroundColor: "rgba(230,126,34,0.08)", fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2 },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                animation: { duration: 0 },
                scales: {
                    x: { ticks: { color: "#5c6c85", maxTicksLimit: 8 }, grid: { color: "rgba(36,48,74,0.4)" } },
                    y: { beginAtZero: true, ticks: { color: "#5c6c85" }, grid: { color: "rgba(36,48,74,0.4)" }, title: { display: true, text: "累计请求（相对页面打开）", color: "#5c6c85" } },
                },
                plugins: { legend: { labels: { color: "#8b9bb4" } } },
            },
        });
    } else {
        requestsChart.data.labels = series.t;
        requestsChart.data.datasets[0].data = series.v1;
        requestsChart.data.datasets[1].data = series.v2;
        requestsChart.update("none");
    }
}

// ── A/B 评测面板 ─────────────────────────
async function refreshEvals() {
    const data = await fetchJSON(API.evals);
    // 数据库为空时回填实测快照（面板顶部显示来源标签）
    let list = data.evals || [];
    $("#eval-source").hidden = list.length > 0;
    if (!list.length) list = [EVAL_SNAPSHOT];
    const tbody = $("#table-eval tbody");

    tbody.innerHTML = list.map((e) => {
        const rec = e.recommendation || "-";
        const recCls = rec === "rollback" ? "state-ERROR" : rec === "advance" ? "state-ACTIVE" : "state-SLEEPING";
        return `<tr>
            <td class="mono">${e.id?.slice(0, 16)}…</td>
            <td>${e.sample_count ?? "-"}</td>
            <td>${e.score_v1_mean?.toFixed(4) ?? "-"}</td>
            <td>${e.score_v2_mean?.toFixed(4) ?? "-"}</td>
            <td class="mono">${e.p_value != null ? e.p_value.toFixed(4) : "-"}</td>
            <td>${e.significant ? "✅ 是" : "否"}</td>
            <td><span class="state-badge ${recCls}">${rec}</span></td>
            <td>${fmtTime(e.created_at)}</td>
        </tr>`;
    }).join("") || `<tr><td colspan="8" style="text-align:center;color:var(--text-dim)">暂无评测记录</td></tr>`;

    if (!list.length) return;
    const latest = list[0]; // 已按时间倒序

    // 统计卡片
    const p = latest.p_value;
    $("#eval-stats").innerHTML = `
        <div class="eval-stat-card"><div class="eval-stat-label">样本量</div><div class="eval-stat-value">${latest.sample_count ?? "-"}</div></div>
        <div class="eval-stat-card"><div class="eval-stat-label">v1 均分</div><div class="eval-stat-value" style="color:${COLORS.v1}">${latest.score_v1_mean?.toFixed(4) ?? "-"}</div></div>
        <div class="eval-stat-card"><div class="eval-stat-label">v2 均分</div><div class="eval-stat-value" style="color:${COLORS.v2}">${latest.score_v2_mean?.toFixed(4) ?? "-"}</div></div>
        <div class="eval-stat-card"><div class="eval-stat-label">p 值</div><div class="eval-stat-value ${p != null && p < 0.05 ? "rollback" : "hold"}">${p != null ? p.toFixed(4) : "-"}</div></div>
        <div class="eval-stat-card"><div class="eval-stat-label">Cohen's d</div><div class="eval-stat-value">${latest.effect_size?.toFixed(4) ?? "-"}</div></div>
        <div class="eval-stat-card"><div class="eval-stat-label">结论</div><div class="eval-stat-value ${latest.recommendation}">${latest.recommendation?.toUpperCase() ?? "-"}</div></div>`;

    // 三维度图 + 解读
    const dims = latest.dimension_scores || {};
    const dv1 = dims.v1 || {};
    const dv2 = dims.v2 || {};
    const dimsArr = ["accuracy", "completeness", "safety"];
    renderDimsChart(
        dimsArr.map((k) => dv1[k] ?? 0),
        dimsArr.map((k) => dv2[k] ?? 0)
    );
    $("#eval-detail").style.display = "";
    renderVerdict(latest);
}

function renderDimsChart(v1Data, v2Data) {
    if (!dimsChart) {
        dimsChart = new Chart($("#chart-dims"), {
            type: "bar",
            data: {
                labels: ["Accuracy 准确性", "Completeness 完整性", "Safety 安全性"],
                datasets: [
                    { label: "v1 (baseline)", data: v1Data, backgroundColor: "rgba(52,152,219,0.65)", borderRadius: 5 },
                    { label: "v2 (candidate)", data: v2Data, backgroundColor: "rgba(230,126,34,0.65)", borderRadius: 5 },
                ],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: "#8b9bb4" }, grid: { display: false } },
                    y: { min: 0, max: 1, ticks: { color: "#5c6c85" }, grid: { color: "rgba(36,48,74,0.4)" } },
                },
                plugins: { legend: { labels: { color: "#8b9bb4" } } },
            },
        });
    } else {
        dimsChart.data.datasets[0].data = v1Data;
        dimsChart.data.datasets[1].data = v2Data;
        dimsChart.update("none");
    }
}

function renderVerdict(e) {
    const rec = e.recommendation;
    let html = "";
    if (rec === "rollback") {
        html = `<b style="color:var(--red)">v2 统计显著劣于 v1（p=${e.p_value?.toFixed(4)} < 0.05）</b><br><br>
        Judge 三维度对比显示 v2 在准确性与完整性上明显落后，效应量 Cohen's d=${e.effect_size?.toFixed(2)}（大效应）。<br><br>
        <b>系统建议：立即回滚。</b>点击上方灰度控制台的「回滚到 v1」按钮，321ms 内流量全部切回 v1。`;
    } else if (rec === "advance") {
        html = `<b style="color:var(--green)">v2 统计显著优于 v1（p=${e.p_value?.toFixed(4)} < 0.05）</b><br><br>
        <b>系统建议：推进灰度。</b>点击「推进到下一阶段」逐步放量至 100%。`;
    } else {
        html = `<b style="color:var(--yellow)">证据不足（p=${e.p_value?.toFixed(4)} ≥ 0.05）</b><br><br>
        当前样本量（n=${e.sample_count}）下无法判定显著差异。<b>建议延长当前阶段继续收集样本</b>——
        这正是 SPEC 规定最小样本量门槛的原因：小样本下即使效应量大，t-test 也可能不显著。`;
    }
    $("#eval-verdict").innerHTML = html;
}

// ── 主循环 ───────────────────────────────
async function pollAll() {
    // 各任务独立 catch，一个失败不拖垮其他
    await Promise.allSettled([
        refreshStatus(),
        refreshModels().catch(() => {}),
        refreshCanary().catch(() => {}),
        refreshMetrics().catch(() => {}),
        refreshEvals().catch(() => {}),
    ]);
}

pollAll();
setInterval(pollAll, POLL_MS);
