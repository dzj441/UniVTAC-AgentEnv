"use strict";

const state = {
  runs: [],
  detail: null,
  activeRunId: null,
  search: "",
  mode: "all",
  selectedStep: 0,
  lightboxItems: [],
  lightboxIndex: 0,
};

const refs = {
  sidebar: document.querySelector("#sidebar"),
  sidebarScrim: document.querySelector("#sidebar-scrim"),
  mobileMenu: document.querySelector("#mobile-menu"),
  runSearch: document.querySelector("#run-search"),
  runList: document.querySelector("#run-list"),
  refreshRuns: document.querySelector("#refresh-runs"),
  serverStatus: document.querySelector("#server-status"),
  serverDot: document.querySelector(".live-dot"),
  emptyState: document.querySelector("#empty-state"),
  runView: document.querySelector("#run-view"),
  errorState: document.querySelector("#error-state"),
  errorMessage: document.querySelector("#error-message"),
  retryButton: document.querySelector("#retry-button"),
  runCrumbs: document.querySelector("#run-crumbs"),
  runTitle: document.querySelector("#run-title"),
  runKind: document.querySelector("#run-kind"),
  runSubtitle: document.querySelector("#run-subtitle"),
  artifactActions: document.querySelector("#artifact-actions"),
  resultStrip: document.querySelector("#result-strip"),
  metricGrid: document.querySelector("#metric-grid"),
  capabilityRow: document.querySelector("#capability-row"),
  recordingNote: document.querySelector("#recording-note"),
  videoPanel: document.querySelector("#video-panel"),
  episodeVideo: document.querySelector("#episode-video"),
  promptPanel: document.querySelector("#prompt-panel"),
  promptContext: document.querySelector("#prompt-context"),
  profileDetail: document.querySelector("#profile-detail"),
  timelineDescription: document.querySelector("#timeline-description"),
  showAll: document.querySelector("#show-all"),
  showFocus: document.querySelector("#show-focus"),
  previousStep: document.querySelector("#previous-step"),
  nextStep: document.querySelector("#next-step"),
  stepRail: document.querySelector("#step-rail"),
  conversation: document.querySelector("#conversation"),
  lightbox: document.querySelector("#lightbox"),
  lightboxClose: document.querySelector("#lightbox-close"),
  lightboxPrevious: document.querySelector("#lightbox-previous"),
  lightboxNext: document.querySelector("#lightbox-next"),
  lightboxImage: document.querySelector("#lightbox-image"),
  lightboxTitle: document.querySelector("#lightbox-title"),
  lightboxCounter: document.querySelector("#lightbox-counter"),
};

const modalityLabels = {
  head_rgb: "Head RGB",
  wrist_rgb: "Wrist RGB",
  left_tactile_marker: "Left tactile",
  right_tactile_marker: "Right tactile",
  left_tactile_rgb: "Left tactile",
  right_tactile_rgb: "Right tactile",
  head_depth: "Head metric depth",
  wrist_depth: "Wrist metric depth",
};

const modalityOrder = [
  "head_rgb",
  "wrist_rgb",
  "left_tactile_marker",
  "right_tactile_marker",
  "left_tactile_rgb",
  "right_tactile_rgb",
  "head_depth",
  "wrist_depth",
];

function modalityLabel(name) {
  if (modalityLabels[name]) return modalityLabels[name];
  return name
    .replaceAll("_manipulated_object_", " · manipulated object · ")
    .replaceAll("_goal_fixture_", " · goal fixture · ")
    .replaceAll("_", " ");
}

function profilePrefix(run) {
  return ["pull_out_key", "put_bottle_in_shelf"].includes(run.task) ? "P" : "L";
}

function node(tag, options = {}, children = []) {
  const element = document.createElement(tag);
  if (options.className) element.className = options.className;
  if (options.text !== undefined) element.textContent = String(options.text);
  if (options.title) element.title = options.title;
  if (options.type) element.type = options.type;
  if (options.href) element.href = options.href;
  if (options.target) element.target = options.target;
  if (options.rel) element.rel = options.rel;
  if (options.role) element.setAttribute("role", options.role);
  if (options.ariaLabel) element.setAttribute("aria-label", options.ariaLabel);
  if (options.onClick) element.addEventListener("click", options.onClick);
  for (const child of Array.isArray(children) ? children : [children]) {
    if (child === null || child === undefined) continue;
    element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return element;
}

function apiUrl(endpoint, params = {}) {
  const base = new URL(".", window.location.href);
  const url = new URL(endpoint, base);
  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined && value !== "") {
      url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

function artifactUrl(path, { download = false } = {}) {
  return apiUrl("api/artifact", {
    run: state.activeRunId,
    path,
    download: download ? 1 : null,
  });
}

async function fetchJson(endpoint, params = {}) {
  const response = await fetch(apiUrl(endpoint, params), {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function cleanSummary(value) {
  return String(value ?? "")
    .replaceAll("**", "")
    .replace(/^#+\s*/gm, "")
    .trim();
}

function formatDuration(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value)) return "—";
  if (value < 60) return `${value.toFixed(value < 10 ? 1 : 0)} s`;
  const minutes = Math.floor(value / 60);
  const rest = Math.round(value % 60);
  return `${minutes}m ${rest}s`;
}

function formatCount(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return new Intl.NumberFormat("zh-CN", { notation: number > 99999 ? "compact" : "standard" }).format(number);
}

function hasFiniteValue(value) {
  return value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
}

function formatDate(value) {
  if (!value) return "时间未知";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function formatSigned(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  const sign = number > 0 ? "+" : "";
  return `${sign}${number.toFixed(digits)}`;
}

function prettyJson(value) {
  return JSON.stringify(value ?? {}, null, 2);
}

function outcomeClass(run) {
  if (run.kind !== "codex") return "capture";
  if (run.valid_for_scoring === false || ["failed", "invalid"].includes(run.status)) {
    return "invalid";
  }
  if (run.outcome?.official_task_success === true) return "success";
  if (run.outcome?.official_task_success === false) return "failure";
  return "pending";
}

function outcomeLabel(run) {
  const result = outcomeClass(run);
  if (result === "success") return "task success";
  if (result === "failure") return "task failed";
  if (result === "invalid") return "invalid run";
  if (result === "capture") return "sensor capture";
  return "in progress";
}

function stepFailed(step) {
  return step.success !== true || step.simulator_execution_succeeded === false;
}

function closeSidebar() {
  refs.sidebar.classList.remove("is-open");
  refs.sidebarScrim.classList.remove("is-open");
}

function setOnline(online, text) {
  refs.serverDot.classList.toggle("is-online", online);
  refs.serverStatus.textContent = text;
}

function setPageState(name, message = "") {
  refs.emptyState.classList.toggle("is-hidden", name !== "empty");
  refs.runView.classList.toggle("is-hidden", name !== "run");
  refs.errorState.classList.toggle("is-hidden", name !== "error");
  if (name === "error") refs.errorMessage.textContent = message;
}

function matchesSearch(run) {
  if (!state.search) return true;
  const haystack = [
    run.id,
    run.name,
    run.group,
    run.profile,
    run.start_condition,
    run.model,
    run.task,
    run.icl,
    run.bbox ? "bbox" : "",
    run.mask ? "mask" : "",
    `level ${run.level}`,
    outcomeLabel(run),
    run.outcome?.predicted_class,
    run.outcome?.true_class,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return haystack.includes(state.search.toLowerCase());
}

function renderRunList() {
  refs.runList.replaceChildren();
  const filtered = state.runs.filter(matchesSearch);
  if (!filtered.length) {
    refs.runList.append(
      node("div", {
        className: "sidebar-empty",
        text: state.search ? "没有匹配的运行记录。" : "agent_runs 中还没有可回放记录。",
      }),
    );
    return;
  }

  let previousGroup = null;
  for (const run of filtered) {
    const group = run.group || "ungrouped";
    if (group !== previousGroup) {
      refs.runList.append(node("div", { className: "run-group-label", text: group }));
      previousGroup = group;
    }
    const resultClass = outcomeClass(run);
    const button = node("button", {
      className: `run-item${run.id === state.activeRunId ? " is-active" : ""}`,
      type: "button",
      onClick: () => loadRun(run.id),
    });
    const title = node("div", { className: "run-item-title", text: run.name });
    const badge = node("span", {
      className: "level-badge",
      text: run.level ? `${profilePrefix(run)}${run.level}` : "CAP",
    });
    button.append(node("div", { className: "run-item-top" }, [title, badge]));
    const result = node("span", {
      className: `run-item-result ${resultClass}`,
      text: outcomeLabel(run),
    });
    const countParts = [];
    if (hasFiniteValue(run.step_eef_count)) {
      countParts.push(`${run.step_eef_count} steps`);
    }
    countParts.push(`${run.tool_count} calls`, `${run.observation_count} obs`);
    if (run.simulator_execution_failure_count) {
      countParts.push(`${run.simulator_execution_failure_count} sim fails`);
    }
    const count = node("span", { text: countParts.join(" · ") });
    button.append(node("div", { className: "run-item-meta" }, [result, count]));
    refs.runList.append(button);
  }
}

async function loadRuns({ preserveSelection = false } = {}) {
  refs.refreshRuns.classList.add("is-spinning");
  try {
    const payload = await fetchJson("api/runs");
    state.runs = payload.runs || [];
    setOnline(true, `${state.runs.length} 条记录 · 可实时刷新`);
    renderRunList();

    if (!state.runs.length) {
      setPageState("empty");
      return;
    }
    const requested = new URLSearchParams(window.location.search).get("run");
    const stillExists = state.runs.some((run) => run.id === state.activeRunId);
    const requestedExists = state.runs.some((run) => run.id === requested);
    if (preserveSelection && stillExists) return;
    const preferred = requestedExists
      ? requested
      : state.runs.find((run) => run.kind === "codex")?.id || state.runs[0].id;
    await loadRun(preferred);
  } catch (error) {
    setOnline(false, "记录服务不可用");
    setPageState("error", error.message);
  } finally {
    refs.refreshRuns.classList.remove("is-spinning");
  }
}

async function loadRun(runId) {
  if (!runId) return;
  state.activeRunId = runId;
  state.selectedStep = 0;
  renderRunList();
  closeSidebar();
  setOnline(true, "正在读取 interaction trace…");
  try {
    const detail = await fetchJson("api/run", { run: runId });
    if (state.activeRunId !== runId) return;
    state.detail = detail;
    state.selectedStep = Math.min(
      Number(new URLSearchParams(window.location.search).get("step")) || 0,
      Math.max(0, detail.steps.length - 1),
    );
    updateUrl();
    renderDetail();
    setPageState("run");
    setOnline(true, `${state.runs.length} 条记录 · 当前 ${detail.steps.length} 个动作`);
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (error) {
    setOnline(false, "读取失败");
    setPageState("error", error.message);
  }
}

function updateUrl() {
  const url = new URL(window.location.href);
  url.searchParams.set("run", state.activeRunId);
  if (state.selectedStep) url.searchParams.set("step", String(state.selectedStep));
  else url.searchParams.delete("step");
  window.history.replaceState({}, "", url);
}

function metric(label, value, title = "") {
  const valueNode = node("span", { className: "metric-value", text: value, title: title || String(value) });
  return node("div", { className: "metric" }, [
    node("span", { className: "metric-label", text: label }),
    valueNode,
  ]);
}

function resultChip(text, tone = "info") {
  return node("span", { className: `result-chip ${tone}`, text });
}

function renderDetail() {
  const detail = state.detail;
  const run = detail.summary;
  refs.runCrumbs.textContent = run.id;
  refs.runTitle.textContent = run.level
    ? `${run.name} · ${profilePrefix(run) === "P" ? "Profile" : "Level"} ${run.level}`
    : run.name;
  refs.runKind.textContent = run.kind === "codex" ? "CODEX RUN" : "ENV CAPTURE";
  refs.runSubtitle.textContent = `${run.task} · ${run.start_condition || "legacy start"} · ${run.profile || "unknown profile"} · ${formatDate(run.created_utc)}`;

  renderArtifactActions(detail.artifacts);
  renderResultStrip(run, detail.outcome);
  renderMetrics(run, detail.runtime);
  renderCapabilities(detail.profile);
  refs.recordingNote.textContent = detail.recording_note;
  renderVideo(detail.artifacts);
  renderPrompt(detail.prompt_context, detail.task_prompt, detail.profile, run.kind);
  refs.timelineDescription.textContent =
    run.kind === "codex"
      ? "每个 Agent 气泡按真实顺序展示动作前公开的 reasoning summary、命令、代码/文件、MCP、子 Agent、网络、图像与消息；紧随其后的 ENV 气泡展示实际反馈和新 observation。"
      : "这是一条环境采集记录：保留实际命令和传感器变化，但没有 Codex App Server activity stream。";
  renderTimelineControls();
  renderConversation();
}

function renderArtifactActions(artifacts) {
  refs.artifactActions.replaceChildren();
  for (const artifact of artifacts.filter((item) => item.type !== "video").slice(0, 4)) {
    refs.artifactActions.append(
      node("a", {
        className: "artifact-link",
        text: artifact.label,
        href: artifactUrl(artifact.path),
        target: "_blank",
        rel: "noreferrer",
      }),
    );
  }
}

function renderResultStrip(run, outcome) {
  refs.resultStrip.replaceChildren();
  refs.resultStrip.append(
    resultChip(`${profilePrefix(run) === "P" ? "Profile" : "Level"} ${run.level ?? "—"}`, "info"),
  );
  if (run.start_condition) {
    refs.resultStrip.append(resultChip(`Start · ${run.start_condition}`, "info"));
  }
  if (run.valid_for_scoring !== null && run.valid_for_scoring !== undefined) {
    refs.resultStrip.append(
      resultChip(run.valid_for_scoring ? "Scoring trace 有效" : "不可计分", run.valid_for_scoring ? "good" : "bad"),
    );
  }
  if (run.icl) {
    refs.resultStrip.append(
      resultChip(`ICL · ${run.icl}`, run.icl === "fixed_demo" ? "good" : "info"),
    );
  }
  if (run.bbox) refs.resultStrip.append(resultChip("BBox · initial only", "info"));
  if (run.mask) refs.resultStrip.append(resultChip("Mask · initial only", "info"));
  if (outcome.predicted_class) {
    const correct = outcome.classification_correct === true;
    refs.resultStrip.append(
      resultChip(
        `分类 ${outcome.predicted_class} / truth ${outcome.true_class ?? "?"}`,
        correct ? "good" : "bad",
      ),
    );
  }
  if (outcome.committed_target) {
    refs.resultStrip.append(
      resultChip(
        `目标 ${outcome.committed_target} / expected ${outcome.expected_target ?? "?"}`,
        outcome.committed_pad_correct ? "good" : "bad",
      ),
    );
  }
  if (outcome.official_task_success !== null && outcome.official_task_success !== undefined) {
    refs.resultStrip.append(
      resultChip(
        `Official success · ${outcome.official_task_success}`,
        outcome.official_task_success ? "good" : "bad",
      ),
    );
  }
}

function renderMetrics(run, runtime) {
  refs.metricGrid.replaceChildren(
    metric("MODEL", run.model || (run.kind === "codex" ? "unknown" : "no agent")),
    metric("EFFORT", run.effort || "—"),
    metric("WALL TIME", formatDuration(run.wall_seconds)),
    metric("TOOL CALLS", run.tool_count ?? "—"),
    metric("OBSERVATIONS", run.observation_count ?? "—"),
    metric("TOKENS", formatCount(run.total_tokens)),
  );
  if (hasFiniteValue(run.step_eef_count)) {
    refs.metricGrid.append(metric("EEF STEPS", run.step_eef_count));
  }
  refs.metricGrid.append(
    metric("SIM EXEC FAILS", run.simulator_execution_failure_count ?? 0),
  );
  if (run.tool_failure_count) {
    refs.metricGrid.append(metric("TOOL FAILS", run.tool_failure_count));
  }
  if (runtime?.event_audit?.passed !== undefined) {
    refs.metricGrid.append(
      metric("EVENT AUDIT", runtime.event_audit.passed ? "passed" : "failed"),
    );
  }
}

function renderCapabilities(profile = {}) {
  refs.capabilityRow.replaceChildren();
  for (const item of profile.public_modalities || []) {
    refs.capabilityRow.append(node("span", { className: "capability-chip", text: item }));
  }
  if (profile.metric_depth) {
    refs.capabilityRow.append(node("span", { className: "capability-chip", text: "metric depth" }));
  }
  if (profile.camera_intrinsics) {
    refs.capabilityRow.append(node("span", { className: "capability-chip", text: "camera intrinsics" }));
  }
  if (profile.camera_extrinsics) {
    refs.capabilityRow.append(node("span", { className: "capability-chip", text: "camera extrinsics" }));
  }
  if (profile.task_success_during_episode === false) {
    refs.capabilityRow.append(node("span", { className: "capability-chip", text: "success terminal-only" }));
  } else if (profile.expose_task_success_after_prediction) {
    refs.capabilityRow.append(node("span", { className: "capability-chip", text: "post-commit success" }));
  }
  if (!refs.capabilityRow.children.length) {
    refs.capabilityRow.append(node("span", { className: "capability-chip", text: "capability metadata unavailable" }));
  }
}

function renderVideo(artifacts) {
  const video = artifacts.find((item) => item.type === "video");
  refs.videoPanel.classList.toggle("is-hidden", !video);
  refs.episodeVideo.pause();
  refs.episodeVideo.removeAttribute("src");
  refs.episodeVideo.load();
  if (video) {
    refs.episodeVideo.src = artifactUrl(video.path);
    refs.episodeVideo.load();
  }
}

function renderPrompt(promptContext, legacyTaskPrompt, profile, kind) {
  refs.promptContext.replaceChildren();
  const sections = Array.isArray(promptContext?.sections)
    ? promptContext.sections
    : [{ role: "operator", label: "Operator / task prompt", text: legacyTaskPrompt }];
  const hasPrompt = sections.some((section) => typeof section.text === "string" && section.text.length);
  if (hasPrompt) {
    for (const section of sections) {
      const card = node("section", { className: `prompt-section prompt-${section.role || "unknown"}` });
      card.append(node("h4", { text: section.label || section.role || "Prompt" }));
      card.append(node("pre", {
        text: typeof section.text === "string" && section.text.length
          ? section.text
          : "未设置",
        className: section.text ? "" : "is-empty",
      }));
      refs.promptContext.append(card);
    }
    if (promptContext?.note) {
      refs.promptContext.append(node("p", { className: "prompt-scope-note", text: promptContext.note }));
    }
  } else {
    refs.promptContext.append(node("p", {
      className: "prompt-empty",
      text: kind === "capture"
        ? "该运行由标准化采集脚本产生，没有 Codex prompt。"
        : "本条记录没有落盘 Benchmark prompt。",
    }));
  }
  refs.profileDetail.replaceChildren();
  const card = node("div", { className: "profile-card" });
  card.append(node("h4", { text: profile.name || "Capability profile" }));
  card.append(node("p", { text: profile.description || "没有 profile 描述。" }));
  const list = node("ul");
  const modalities = profile.public_modalities || [];
  const tactile = profile.expose_tactile ?? modalities.some((item) => item.includes("tactile"));
  list.append(node("li", { text: `触觉：${tactile ? "开放" : "不开放"}` }));
  const terminalOnly = profile.task_success_during_episode === false;
  const legacySuccess = profile.expose_task_success_after_prediction;
  list.append(node("li", {
    text: `Success feedback：${terminalOnly ? "仅 finish 后公开" : legacySuccess ? "commit 后开放" : "不开放"}`,
  }));
  if (profile.metric_depth !== undefined) {
    list.append(node("li", { text: `米制深度：${profile.metric_depth ? "开放" : "不开放"}` }));
  }
  if (profile.camera_intrinsics !== undefined || profile.camera_extrinsics !== undefined) {
    list.append(node("li", {
      text: `相机标定：内参 ${profile.camera_intrinsics ? "开放" : "关闭"} / 外参 ${profile.camera_extrinsics ? "开放" : "关闭"}`,
    }));
  }
  list.append(node("li", { text: `基础机器人状态：${(profile.public_robot_state || []).length} 项` }));
  card.append(list);
  refs.profileDetail.append(card);
}

function renderTimelineControls() {
  refs.stepRail.replaceChildren();
  const steps = state.detail.steps;
  for (const [index, step] of steps.entries()) {
    const button = node("button", {
      className: `step-dot${index === state.selectedStep ? " is-active" : ""}${stepFailed(step) ? " is-failed" : ""}`,
      text: `A${index + 1}`,
      type: "button",
      title: `${step.tool_label} · ${step.prior_observation_id || "无输入 observation"}`,
      onClick: () => selectStep(index, true),
    });
    button.setAttribute("aria-current", index === state.selectedStep ? "step" : "false");
    refs.stepRail.append(button);
  }
  refs.previousStep.disabled = state.selectedStep <= 0;
  refs.nextStep.disabled = state.selectedStep >= steps.length - 1;
  refs.showAll.classList.toggle("is-active", state.mode === "all");
  refs.showFocus.classList.toggle("is-active", state.mode === "focus");
  requestAnimationFrame(() => {
    refs.stepRail.children[state.selectedStep]?.scrollIntoView({ inline: "center", block: "nearest" });
  });
}

function selectStep(index, scroll = false) {
  const max = Math.max(0, (state.detail?.steps.length || 1) - 1);
  state.selectedStep = Math.min(max, Math.max(0, index));
  updateUrl();
  renderTimelineControls();
  if (state.mode === "focus") {
    renderConversation();
  } else if (scroll) {
    document.querySelector(`#round-${state.selectedStep}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function setMode(mode) {
  state.mode = mode;
  renderTimelineControls();
  renderConversation();
}

function renderConversation() {
  refs.conversation.replaceChildren();
  const steps = state.detail.steps;
  if (!steps.length) {
    refs.conversation.append(
      node("div", { className: "sidebar-empty", text: "这条记录还没有完整工具调用。刷新后可继续查看。" }),
    );
    return;
  }

  const selected = state.mode === "focus" ? [steps[state.selectedStep]] : steps;
  for (const step of selected) {
    const block = node("article", { className: "round-block" });
    block.id = `round-${step.index}`;
    block.append(
      node("div", {
        className: "round-marker",
        text: `ROUND ${String(step.index + 1).padStart(2, "0")}`,
      }),
    );
    if (state.mode === "focus" && step.input_observation) {
      block.append(renderEnvironmentRow(step, { inputContext: true }));
    }
    block.append(renderAgentRow(step));
    block.append(renderEnvironmentRow(step));
    refs.conversation.append(block);
  }

  if (state.mode === "all" && (state.detail.tail_agent_activity || []).length) {
    refs.conversation.append(renderTailActivity(state.detail.tail_agent_activity));
  }
}

function bubbleHeader(speaker, title, context) {
  const avatar = node("span", { className: "speaker-avatar", text: speaker === "ENV" ? "E" : "A" });
  const speakerBlock = node("div", { className: "speaker" }, [
    avatar,
    node("span", { className: "speaker-title", text: `${speaker} · ${title}` }),
  ]);
  return node("header", { className: "bubble-header" }, [
    speakerBlock,
    node("span", { className: "speaker-context", text: context }),
  ]);
}

function renderAgentRow(step) {
  const body = node("div", { className: "bubble-body" });
  if ((step.agent_activity || []).length) body.append(renderAgentActivity(step.agent_activity));

  body.append(renderAction(step));
  const context = `${step.prior_observation_id ? `基于 ${step.prior_observation_id}` : "无 observation"} · +${formatDuration(step.elapsed_seconds)}`;
  const bubble = node("div", { className: "chat-bubble" }, [
    bubbleHeader(
      state.detail.summary.kind === "codex" ? "AGENT" : "SCRIPT",
      step.tool_label,
      context,
    ),
    body,
  ]);
  return node("div", { className: "chat-row agent" }, bubble);
}

function activityFailed(event) {
  const status = String(event.status || "").toLowerCase();
  const exitCode = event.details?.exitCode;
  return ["error", "failed", "declined", "cancelled"].some((item) => status.includes(item))
    || (Number.isFinite(Number(exitCode)) && Number(exitCode) !== 0);
}

function renderAgentActivity(events, label = "OBSERVABLE AGENT ACTIVITY") {
  const section = node("section", { className: "agent-activity" });
  section.append(
    node("div", { className: "activity-section-heading" }, [
      node("span", { className: "section-label", text: label }),
      node("span", { className: "activity-count", text: `${events.length} events` }),
    ]),
  );
  const list = node("div", { className: "activity-list" });
  for (const event of events) {
    const failed = activityFailed(event);
    const card = node("article", {
      className: `activity-card activity-${event.kind || "unknown"}${failed ? " failed" : ""}`,
    });
    const metadata = [];
    if (event.status) metadata.push(event.status);
    if (hasFiniteValue(event.elapsed_seconds)) metadata.push(`+${formatDuration(event.elapsed_seconds)}`);
    card.append(
      node("header", { className: "activity-header" }, [
        node("span", { className: "activity-kind", text: event.label || event.kind || "Agent activity" }),
        node("span", { className: "activity-meta", text: metadata.join(" · ") }),
      ]),
    );
    if (event.title) {
      card.append(node("pre", { className: "activity-title", text: event.title }));
    }
    for (const part of event.parts || []) {
      card.append(
        node("p", {
          className: "activity-part",
          text: event.kind === "reasoning" ? cleanSummary(part) : part,
        }),
      );
    }
    if (event.details && typeof event.details === "object") {
      const details = node("details", { className: "raw-details activity-details" });
      details.append(node("summary", { text: "完整公开事件数据" }));
      details.append(node("pre", { text: prettyJson(event.details) }));
      card.append(details);
    }
    list.append(card);
  }
  section.append(list);
  return section;
}

function actionChips(step) {
  const args = step.arguments || {};
  const chips = [];
  if (step.tool === "probe_gripper") {
    chips.push(`Δ gripper ${formatSigned(Number(args.delta_gripper) * 1000, 2)} mm`);
  } else if (step.tool === "commit_classification") {
    chips.push(`${args.predicted_class ?? "?"} → ${args.target_pad ?? "?"}`);
  } else if (step.tool === "act_delta_ee" || step.tool === "step_eef") {
    if (Array.isArray(args.delta_position)) {
      chips.push(`ΔXYZ [${args.delta_position.map((item) => formatSigned(Number(item) * 100, 1)).join(", ")}] cm`);
    }
    if (Array.isArray(args.delta_rpy) && args.delta_rpy.some((item) => Number(item) !== 0)) {
      chips.push(`ΔRPY [${args.delta_rpy.map((item) => formatSigned(item, 2)).join(", ")}] rad`);
    }
    if (Number(args.delta_gripper)) {
      chips.push(`Δ gripper ${formatSigned(Number(args.delta_gripper) * 1000, 2)} mm`);
    } else {
      chips.push("gripper hold");
    }
  } else if (step.tool === "wait_physics") {
    chips.push(`${args.steps ?? "?"} physics steps`);
  } else if (step.tool === "finish_episode") {
    chips.push("terminal evaluation");
  } else if (step.tool === "start_episode") {
    chips.push("reset + first observation");
  } else if (step.tool === "sam3_segment") {
    chips.push(`${args.camera ?? "?"} · ${args.mode ?? "?"}`);
    if (args.prompt) chips.push(`prompt: ${args.prompt}`);
  } else if (step.tool === "estimate_metric_depth") {
    chips.push(`${args.camera ?? "?"} · resolution ${args.resolution_level ?? 4}`);
    if (Array.isArray(args.sample_points)) chips.push(`${args.sample_points.length} samples`);
  }
  return chips;
}

function renderAction(step) {
  const card = node("section", { className: "action-card" });
  const failed = stepFailed(step);
  const statusText = !step.success
    ? "× tool failed"
    : step.simulator_execution_succeeded === false
      ? `✓ host accepted · ⚠ sim execution failed · ${formatDuration(step.duration_seconds)}`
      : `✓ host accepted · ${formatDuration(step.duration_seconds)}`;
  const status = node("span", {
    className: `action-status${failed ? " failed" : ""}`,
    text: statusText,
  });
  card.append(
    node("div", { className: "action-heading" }, [
      node("strong", { text: step.tool }),
      status,
    ]),
  );
  const chips = node("div", { className: "action-chips" });
  for (const text of actionChips(step)) chips.append(node("span", { className: "action-chip", text }));
  if (!chips.children.length) chips.append(node("span", { className: "action-chip", text: "read-only call" }));
  card.append(chips);
  const target = step.execution_target || "simulator";
  const details = node("details", { className: "raw-details" });
  const targetLabel = target === "perception" ? "宿主感知服务请求" : target === "rejected" ? "宿主拒绝（未发送）" : "实际发送给仿真的命令";
  details.append(node("summary", { text: `${targetLabel} · raw JSON` }));
  details.append(node("pre", { text: prettyJson(step.backend_request ?? step.simulator_command) }));
  card.append(details);
  return card;
}

function renderEnvironmentRow(step, { inputContext = false } = {}) {
  if (inputContext) {
    const body = node("div", { className: "bubble-body" });
    body.append(renderObservation(step.input_observation, "本轮输入 Observation"));
    const bubble = node("div", { className: "input-context" }, [
      bubbleHeader("ENV", "Agent 本轮看到的输入", step.prior_observation_id || "—"),
      body,
    ]);
    return node("div", { className: "chat-row env" }, bubble);
  }

  const body = node("div", { className: "bubble-body" });
  body.append(renderFeedback(step));
  if (Array.isArray(step.derived_artifacts) && step.derived_artifacts.length) {
    body.append(renderDerivedArtifacts(step.derived_artifacts));
  }
  if (Object.keys(step.state_delta || {}).length) body.append(renderStateDelta(step.state_delta));
  if (step.output_observation && step.fresh_observation) {
    body.append(renderObservation(step.output_observation, "动作后的新 Observation"));
  } else {
    const unchanged = node("section", { className: "no-observation" });
    unchanged.append(node("span", { className: "section-label", text: "OBSERVATION TRANSITION" }));
    unchanged.append(
      node("p", {
        text: step.next_observation_id
          ? `本次调用没有刷新传感器；当前 observation 仍为 ${step.next_observation_id}。`
          : "本次调用没有返回新的传感器 observation。",
      }),
    );
    body.append(unchanged);
  }
  const status = step.simulator_execution_succeeded === false
    ? `${step.environment_response?.status || "action"} · execution failed`
    : step.environment_response?.status || (step.success ? "accepted" : "failed");
  const context = `${step.prior_observation_id || "—"} → ${step.next_observation_id || "—"}`;
  const bubble = node("div", { className: "chat-bubble" }, [
    bubbleHeader("ENV", status, context),
    body,
  ]);
  return node("div", { className: "chat-row env" }, bubble);
}

function renderFeedback(step) {
  const response = step.environment_response || {};
  const feedback = response.feedback && typeof response.feedback === "object" ? response.feedback : {};
  const block = node("section", { className: "feedback-block" });
  const top = node("div", { className: "feedback-top" });
  const content = node("div");
  content.append(node("span", {
    className: "section-label",
    text: step.execution_target === "perception" ? "PERCEPTION RESULT" : "SIMULATOR FEEDBACK",
  }));
  const parts = [];
  const executionSucceeded = feedback.execution_succeeded ?? response.execution_succeeded;
  if (executionSucceeded !== undefined) parts.push(`execution_succeeded=${executionSucceeded}`);
  if (feedback.task_success !== undefined) parts.push(`task_success=${feedback.task_success}`);
  if (response.remaining_probes !== undefined) parts.push(`remaining_probes=${response.remaining_probes}`);
  if (response.predicted_class) parts.push(`committed ${response.predicted_class} → ${response.committed_target}`);
  content.append(
    node("p", {
      text: parts.length ? parts.join(" · ") : "环境已返回协议状态；展开 raw response 可查看完整公开字段。",
    }),
  );
  top.append(content);
  top.append(
    node("span", {
      className: `feedback-status${stepFailed(step) ? " failed" : ""}`,
      text: executionSucceeded === false
        ? "execution failed"
        : response.status || (step.success ? "accepted" : "failed"),
    }),
  );
  block.append(top);
  const details = node("details", { className: "raw-details" });
  details.append(node("summary", { text: "环境公开反馈 · raw JSON" }));
  details.append(node("pre", { text: prettyJson(response) }));
  block.append(details);
  return block;
}

function renderDerivedArtifacts(artifacts) {
  const block = node("section", { className: "observation-block" });
  block.append(node("span", { className: "section-label", text: "MODEL-DERIVED ARTIFACTS" }));
  const lightboxItems = artifacts.map((item) => ({
    src: artifactUrl(item.artifact),
    title: item.label,
  }));
  const grid = node("div", { className: "sensor-grid" });
  artifacts.forEach((item, index) => {
    const image = node("img");
    image.src = artifactUrl(item.artifact);
    image.alt = item.label;
    image.loading = "lazy";
    image.decoding = "async";
    grid.append(node("button", {
      className: "sensor-card",
      type: "button",
      ariaLabel: `放大 ${item.label}`,
      onClick: () => openLightbox(lightboxItems, index),
    }, [image, node("span", { className: "sensor-label", text: item.label })]));
  });
  block.append(grid);
  return block;
}

function renderStateDelta(delta) {
  const section = node("section", { className: "state-delta" });
  section.append(node("span", { className: "section-label", text: "ACTUAL STATE CHANGE" }));
  const chips = node("div", { className: "state-chips" });
  if (Number.isFinite(Number(delta.gripper_delta_mm))) {
    chips.append(
      node("span", {
        className: "state-chip",
        text: `actual Δgripper ${formatSigned(delta.gripper_delta_mm, 3)} mm`,
      }),
    );
  }
  if (Array.isArray(delta.end_effector_delta_xyz_m)) {
    chips.append(
      node("span", {
        className: "state-chip",
        text: `actual ΔXYZ [${delta.end_effector_delta_xyz_m
          .map((item) => formatSigned(Number(item) * 100, 2))
          .join(", ")}] cm`,
      }),
    );
  }
  if (Number.isFinite(Number(delta.end_effector_translation_m))) {
    chips.append(
      node("span", {
        className: "state-chip",
        text: `translation ${(Number(delta.end_effector_translation_m) * 100).toFixed(2)} cm`,
      }),
    );
  }
  section.append(chips);
  return section;
}

function renderObservation(observation, title) {
  const block = node("section", { className: "observation-block" });
  const obsTitle = node("div", { className: "observation-title" }, [
    node("strong", { text: observation.observation_id }),
    node("span", { text: observation.stage || "unknown stage" }),
  ]);
  const legacyCounts = observation.probe_count !== null && observation.probe_count !== undefined;
  const counts = node("span", {
    className: "observation-counts",
    text: legacyCounts
      ? `probe ${observation.probe_count ?? "—"} · action ${observation.post_prediction_action_count ?? "—"}`
      : observation.profile || "generic embodied observation",
  });
  block.append(node("header", { className: "observation-header" }, [obsTitle, counts]));

  const entries = Object.entries(observation.modalities || {}).sort(([a], [b]) => {
    const aIndex = modalityOrder.indexOf(a);
    const bIndex = modalityOrder.indexOf(b);
    if (aIndex === -1 && bIndex === -1) return a.localeCompare(b);
    if (aIndex === -1) return 1;
    if (bIndex === -1) return -1;
    return aIndex - bIndex;
  });
  const lightboxItems = entries.map(([name, meta]) => ({
    src: artifactUrl(meta.artifact),
    title: `${observation.observation_id} · ${modalityLabel(name)}`,
  }));
  if (entries.length) {
    const grid = node("div", { className: "sensor-grid" });
    entries.forEach(([name, metadata], index) => {
      const image = node("img");
      image.src = artifactUrl(metadata.artifact);
      image.alt = `${observation.observation_id} ${modalityLabel(name)}`;
      image.loading = "lazy";
      image.decoding = "async";
      const card = node("button", {
        className: "sensor-card",
        type: "button",
        ariaLabel: `放大 ${modalityLabel(name)}`,
        onClick: () => openLightbox(lightboxItems, index),
      }, [image, node("span", { className: "sensor-label", text: modalityLabel(name) })]);
      grid.append(card);
    });
    block.append(grid);
  }

  const meta = node("div", { className: "observation-meta" });
  meta.append(node("span", { className: "section-label", text: title }));
  const robot = observation.robot_state || {};
  const stateChips = node("div", { className: "state-chips" });
  if (Number.isFinite(Number(robot.gripper_qpos))) {
    stateChips.append(
      node("span", {
        className: "state-chip",
        text: `gripper ${(Number(robot.gripper_qpos) * 1000).toFixed(3)} mm`,
      }),
    );
  }
  if (Number.isFinite(Number(robot.gripper_width_m))) {
    stateChips.append(
      node("span", {
        className: "state-chip",
        text: `gripper width ${(Number(robot.gripper_width_m) * 1000).toFixed(3)} mm`,
      }),
    );
  }
  const endEffectorPose = robot.end_effector_pose_robot_base_wxyz_7d
    || robot.end_effector_pose_robot_base_7d;
  if (Array.isArray(endEffectorPose)) {
    stateChips.append(
      node("span", {
        className: "state-chip",
        text: `EE xyz [${endEffectorPose
          .slice(0, 3)
          .map((item) => Number(item).toFixed(4))
          .join(", ")}] m`,
      }),
    );
  }
  if (observation.task_success !== undefined) {
    stateChips.append(node("span", { className: "state-chip", text: `task_success ${observation.task_success}` }));
  }
  meta.append(stateChips);

  const healthEntries = Object.entries(observation.tactile_health || {});
  if (healthEntries.length) {
    const health = node("div", { className: "tactile-health" });
    for (const [side, value] of healthEntries) {
      health.append(
        node("span", {
          className: `health-chip${value.healthy ? "" : " unhealthy"}`,
          text: `${side} markers ${value.plausible_marker_components ?? "?"}/${value.expected_markers ?? "?"} · ${value.healthy ? "healthy" : "degraded"}`,
        }),
      );
    }
    meta.append(health);
  }

  if (observation.composite) {
    meta.append(
      node("button", {
        className: "composite-link",
        text: "单独查看完整 observation composite ↗",
        type: "button",
        onClick: () =>
          openLightbox(
            [{ src: artifactUrl(observation.composite), title: `${observation.observation_id} · composite` }],
            0,
          ),
      }),
    );
  }
  const details = node("details", { className: "robot-details" });
  details.append(node("summary", { text: "完整基础机器人状态" }));
  details.append(node("pre", { text: prettyJson(robot) }));
  meta.append(details);
  if (Object.keys(observation.camera_calibration || {}).length) {
    const calibration = node("details", { className: "robot-details" });
    calibration.append(node("summary", { text: "相机内外参" }));
    calibration.append(node("pre", { text: prettyJson(observation.camera_calibration) }));
    meta.append(calibration);
  }
  if (Object.keys(observation.annotations || {}).length) {
    const annotations = node("details", { className: "robot-details" });
    annotations.append(node("summary", { text: "匿名 BBox / Mask 元数据" }));
    annotations.append(node("pre", { text: prettyJson(observation.annotations) }));
    meta.append(annotations);
  }
  block.append(meta);
  return block;
}

function renderTailActivity(events) {
  const block = node("article", { className: "round-block tail-message" });
  block.append(node("div", { className: "round-marker", text: "FINAL" }));
  const body = node("div", { className: "bubble-body" });
  body.append(renderAgentActivity(events, "FINAL OBSERVABLE AGENT ACTIVITY"));
  const bubble = node("div", { className: "chat-bubble" }, [
    bubbleHeader("AGENT", "末次机器人调用后的公开活动", "turn completed"),
    body,
  ]);
  block.append(node("div", { className: "chat-row agent" }, bubble));
  return block;
}

function openLightbox(items, index) {
  state.lightboxItems = items;
  state.lightboxIndex = index;
  updateLightbox();
  refs.lightbox.showModal();
}

function updateLightbox() {
  const item = state.lightboxItems[state.lightboxIndex];
  if (!item) return;
  refs.lightboxImage.src = item.src;
  refs.lightboxTitle.textContent = item.title;
  refs.lightboxCounter.textContent = `${state.lightboxIndex + 1} / ${state.lightboxItems.length}`;
  refs.lightboxPrevious.disabled = state.lightboxItems.length <= 1;
  refs.lightboxNext.disabled = state.lightboxItems.length <= 1;
}

function moveLightbox(delta) {
  if (!state.lightboxItems.length) return;
  state.lightboxIndex =
    (state.lightboxIndex + delta + state.lightboxItems.length) % state.lightboxItems.length;
  updateLightbox();
}

refs.runSearch.addEventListener("input", (event) => {
  state.search = event.target.value.trim();
  renderRunList();
});
refs.refreshRuns.addEventListener("click", () => loadRuns({ preserveSelection: true }));
refs.retryButton.addEventListener("click", () => loadRuns());
refs.showAll.addEventListener("click", () => setMode("all"));
refs.showFocus.addEventListener("click", () => setMode("focus"));
refs.previousStep.addEventListener("click", () => selectStep(state.selectedStep - 1, true));
refs.nextStep.addEventListener("click", () => selectStep(state.selectedStep + 1, true));
refs.mobileMenu.addEventListener("click", () => {
  refs.sidebar.classList.add("is-open");
  refs.sidebarScrim.classList.add("is-open");
});
refs.sidebarScrim.addEventListener("click", closeSidebar);
refs.lightboxClose.addEventListener("click", () => refs.lightbox.close());
refs.lightboxPrevious.addEventListener("click", () => moveLightbox(-1));
refs.lightboxNext.addEventListener("click", () => moveLightbox(1));
refs.lightbox.addEventListener("click", (event) => {
  if (event.target === refs.lightbox) refs.lightbox.close();
});

window.addEventListener("keydown", (event) => {
  const target = event.target;
  const typing = target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement;
  if (typing) return;
  if (refs.lightbox.open) {
    if (event.key === "ArrowLeft") moveLightbox(-1);
    if (event.key === "ArrowRight") moveLightbox(1);
    return;
  }
  if (event.key === "ArrowLeft") selectStep(state.selectedStep - 1, true);
  if (event.key === "ArrowRight") selectStep(state.selectedStep + 1, true);
});

window.addEventListener("popstate", () => {
  const runId = new URLSearchParams(window.location.search).get("run");
  if (runId && runId !== state.activeRunId) loadRun(runId);
});

loadRuns();
