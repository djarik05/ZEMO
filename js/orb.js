/**
 * ZEMO INTELLIGENCE ORB
 * Sophisticated, high-performance Canvas 2D organic AI core.
 * Features multi-layered depth, orbital mechanics, particle constellation,
 * and mouse-responsive parallax.
 */

let canvas, ctx;
let animationFrameId;
let width, height, dpr;
let mouseX = 0, mouseY = 0;
let tiltX = 0, tiltY = 0;
let targetTiltX = 0, targetTiltY = 0;
let isHovering = false;

// Global Orb State
let currentState = "idle"; // "idle" | "inspecting" | "simulating" | "ready"
let stateTime = 0;
let rotationAngle = 0;

// Constellation Nodes
const NODE_COUNT = 28;
let nodes = [];

// Micro-particles
const PARTICLE_COUNT = 45;
let particles = [];

// Orbital Rings definitions
const RINGS = [
  { radiusX: 130, radiusY: 48, tilt: -0.35, speed: 0.008, color: "rgba(77, 88, 55, 0.40)" },
  { radiusX: 145, radiusY: 38, tilt: 0.52, speed: -0.006, color: "rgba(158, 172, 149, 0.55)" },
  { radiusX: 160, radiusY: 55, tilt: -0.15, speed: 0.005, color: "rgba(77, 88, 55, 0.30)" },
  { radiusX: 115, radiusY: 65, tilt: 0.85, speed: -0.010, color: "rgba(158, 172, 149, 0.40)" }
];

export function initOrb() {
  canvas = document.getElementById("orb-canvas");
  if (!canvas) return;

  ctx = canvas.getContext("2d");
  setupCanvasDimensions();
  initNodesAndParticles();

  // Mouse Interactivity
  window.addEventListener("resize", setupCanvasDimensions);

  const container = canvas.closest(".orb-container") || canvas;
  container.addEventListener("mousemove", onMouseMove);
  container.addEventListener("mouseenter", () => { isHovering = true; });
  container.addEventListener("mouseleave", onMouseLeave);

  // Check prefers-reduced-motion
  const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  if (mediaQuery.matches) {
    renderStaticOrb();
  } else {
    requestAnimationFrame(renderLoop);
  }

  mediaQuery.addEventListener("change", (e) => {
    if (e.matches) {
      cancelAnimationFrame(animationFrameId);
      renderStaticOrb();
    } else {
      requestAnimationFrame(renderLoop);
    }
  });
}

function setupCanvasDimensions() {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  dpr = window.devicePixelRatio || 1;
  width = rect.width;
  height = rect.height;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
}

function initNodesAndParticles() {
  nodes = [];
  for (let i = 0; i < NODE_COUNT; i++) {
    const phi = Math.acos(-1 + (2 * i) / NODE_COUNT);
    const theta = Math.sqrt(NODE_COUNT * Math.PI) * phi;
    nodes.push({
      baseX: Math.cos(theta) * Math.sin(phi),
      baseY: Math.sin(theta) * Math.sin(phi),
      baseZ: Math.cos(phi),
      x: 0, y: 0, z: 0,
      radius: 1.8 + Math.random() * 1.6,
      phase: Math.random() * Math.PI * 2
    });
  }

  particles = [];
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    particles.push({
      x: (Math.random() - 0.5) * 280,
      y: (Math.random() - 0.5) * 280,
      z: (Math.random() - 0.5) * 140,
      radius: 0.8 + Math.random() * 1.4,
      speedX: (Math.random() - 0.5) * 0.4,
      speedY: (Math.random() - 0.5) * 0.4,
      opacity: 0.2 + Math.random() * 0.5
    });
  }
}

function onMouseMove(e) {
  const rect = canvas.getBoundingClientRect();
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;

  mouseX = (e.clientX - centerX) / (rect.width / 2);
  mouseY = (e.clientY - centerY) / (rect.height / 2);

  targetTiltX = mouseX * 18;
  targetTiltY = mouseY * 18;
}

function onMouseLeave() {
  isHovering = false;
  targetTiltX = 0;
  targetTiltY = 0;
}

export function setOrbState(newState) {
  currentState = newState;
  stateTime = 0;

  const stateBadge = document.getElementById("orb-state-text");
  const stateIndicator = document.querySelector(".orb-state-indicator");

  if (stateBadge) {
    switch (newState) {
      case "idle":
        stateBadge.textContent = "AWAITING DATASET";
        if (stateIndicator) stateIndicator.style.backgroundColor = "var(--sage)";
        break;
      case "inspecting":
        stateBadge.textContent = "INSPECTING DATASET";
        if (stateIndicator) stateIndicator.style.backgroundColor = "var(--warning)";
        break;
      case "simulating":
        stateBadge.textContent = "ANALYSIS IN PROGRESS";
        if (stateIndicator) stateIndicator.style.backgroundColor = "var(--olive)";
        break;
      case "ready":
        stateBadge.textContent = "VERIFIED / READY";
        if (stateIndicator) stateIndicator.style.backgroundColor = "var(--success)";
        break;
    }
  }
}

function renderLoop(timestamp) {
  stateTime += 0.016;
  ctx.clearRect(0, 0, width, height);

  // Smooth Damped Parallax
  tiltX += (targetTiltX - tiltX) * 0.06;
  tiltY += (targetTiltY - tiltY) * 0.06;

  // Autonomous Drift if not hovered
  if (!isHovering) {
    tiltX += Math.sin(stateTime * 0.8) * 0.15;
    tiltY += Math.cos(stateTime * 0.6) * 0.15;
  }

  // Adjust speed by state
  let speedMultiplier = 1.0;
  if (currentState === "inspecting") speedMultiplier = 2.4;
  if (currentState === "simulating") speedMultiplier = 1.8;
  if (currentState === "ready") speedMultiplier = 0.8;

  rotationAngle += 0.007 * speedMultiplier;

  const centerX = width / 2 + tiltX;
  const centerY = height / 2 + tiltY;
  const baseRadius = Math.min(width, height) * 0.28;

  // 1. Draw Ambient Outer Glow
  drawAmbientGlow(centerX, centerY, baseRadius);

  // 2. Draw Back Orbital Rings
  drawOrbitalRings(centerX, centerY, baseRadius, true);

  // 3. Draw Core Sphere
  drawCoreSphere(centerX, centerY, baseRadius);

  // 4. Draw Constellation Network (Nodes + Interconnects)
  drawConstellation(centerX, centerY, baseRadius);

  // 5. Draw Floating Ambient Particles
  drawParticles(centerX, centerY);

  // 6. Draw Front Orbital Rings
  drawOrbitalRings(centerX, centerY, baseRadius, false);

  // 7. Pulse Ring Effect (when simulating or inspecting)
  if (currentState === "inspecting" || currentState === "simulating") {
    drawScanningWave(centerX, centerY, baseRadius);
  }

  animationFrameId = requestAnimationFrame(renderLoop);
}

function drawAmbientGlow(cx, cy, r) {
  const glowRadius = r * 1.8;
  const grad = ctx.createRadialGradient(cx, cy, r * 0.4, cx, cy, glowRadius);

  if (currentState === "ready") {
    grad.addColorStop(0, "rgba(98, 123, 82, 0.35)");
    grad.addColorStop(0.6, "rgba(158, 172, 149, 0.12)");
    grad.addColorStop(1, "rgba(247, 246, 242, 0)");
  } else {
    grad.addColorStop(0, "rgba(158, 172, 149, 0.40)");
    grad.addColorStop(0.5, "rgba(220, 226, 214, 0.25)");
    grad.addColorStop(1, "rgba(247, 246, 242, 0)");
  }

  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.arc(cx, cy, glowRadius, 0, Math.PI * 2);
  ctx.fill();
}

function drawCoreSphere(cx, cy, r) {
  // Main volumetric spherical body
  const grad = ctx.createRadialGradient(
    cx - r * 0.3, cy - r * 0.35, r * 0.1,
    cx, cy, r
  );

  grad.addColorStop(0, "#FAF9F5");
  grad.addColorStop(0.25, "#DCE2D6");
  grad.addColorStop(0.65, "#9EAC95");
  grad.addColorStop(0.92, "#4D5837");
  grad.addColorStop(1, "#353E29");

  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.shadowColor = "rgba(77, 88, 55, 0.22)";
  ctx.shadowBlur = 24;
  ctx.shadowOffsetY = 8;
  ctx.fill();
  ctx.restore();

  // Internal organic swirling contour lines
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r - 1, 0, Math.PI * 2);
  ctx.clip();

  ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
  ctx.lineWidth = 1.2;

  for (let i = -2; i <= 2; i++) {
    const yOffset = i * (r * 0.32);
    const wave = Math.sin(stateTime + i) * 6;

    ctx.beginPath();
    ctx.ellipse(cx, cy + yOffset + wave, r * 0.88, r * 0.22, 0.15, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Specular sheen highlight
  const sheen = ctx.createRadialGradient(
    cx - r * 0.32, cy - r * 0.38, 2,
    cx - r * 0.32, cy - r * 0.38, r * 0.45
  );
  sheen.addColorStop(0, "rgba(255, 255, 255, 0.65)");
  sheen.addColorStop(0.4, "rgba(255, 255, 255, 0.18)");
  sheen.addColorStop(1, "rgba(255, 255, 255, 0)");

  ctx.fillStyle = sheen;
  ctx.beginPath();
  ctx.arc(cx - r * 0.32, cy - r * 0.38, r * 0.45, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

function drawOrbitalRings(cx, cy, r, isBack) {
  const scale = r / 100;

  RINGS.forEach((ring, idx) => {
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(ring.tilt);

    const rx = ring.radiusX * scale;
    const ry = ring.radiusY * scale;

    ctx.beginPath();
    ctx.ellipse(0, 0, rx, ry, 0, isBack ? Math.PI : 0, isBack ? Math.PI * 2 : Math.PI);
    ctx.strokeStyle = ring.color;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 6]);
    ctx.stroke();

    // Orbital marker node traveling on the ring
    const markerAngle = rotationAngle * ring.speed * 120 + idx;
    const markerX = Math.cos(markerAngle) * rx;
    const markerY = Math.sin(markerAngle) * ry;

    // Check depth
    const isMarkerBack = markerY < 0;
    if (isMarkerBack === isBack) {
      ctx.beginPath();
      ctx.arc(markerX, markerY, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(77, 88, 55, 0.85)";
      ctx.fill();

      ctx.beginPath();
      ctx.arc(markerX, markerY, 5, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(158, 172, 149, 0.50)";
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.stroke();
    }

    ctx.restore();
  });
}

function drawConstellation(cx, cy, r) {
  const currentRadius = r * 0.88;

  // Project 3D nodes to 2D
  nodes.forEach(node => {
    // 3D rotation matrix around Y axis
    const cosY = Math.cos(rotationAngle);
    const sinY = Math.sin(rotationAngle);

    const rotX = node.baseX * cosY - node.baseZ * sinY;
    const rotZ = node.baseX * sinY + node.baseZ * cosY;
    const rotY = node.baseY;

    node.x = cx + rotX * currentRadius;
    node.y = cy + rotY * currentRadius;
    node.z = rotZ;
  });

  // Draw Interconnecting Lines
  ctx.lineWidth = 0.8;
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const n1 = nodes[i];
      const n2 = nodes[j];

      // Only connect if both are in front half or within proximity
      if (n1.z > -0.2 && n2.z > -0.2) {
        const dx = n1.x - n2.x;
        const dy = n1.y - n2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 55) {
          const alpha = (1 - dist / 55) * 0.45;
          ctx.strokeStyle = `rgba(77, 88, 55, ${alpha})`;
          ctx.beginPath();
          ctx.moveTo(n1.x, n1.y);
          ctx.lineTo(n2.x, n2.y);
          ctx.stroke();
        }
      }
    }
  }

  // Draw Nodes
  nodes.forEach(node => {
    if (node.z > -0.1) {
      const depthScale = (node.z + 1) * 0.5;
      const nodeAlpha = 0.35 + depthScale * 0.65;

      ctx.beginPath();
      ctx.arc(node.x, node.y, node.radius * depthScale, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(53, 62, 41, ${nodeAlpha})`;
      ctx.fill();
    }
  });
}

function drawParticles(cx, cy) {
  particles.forEach(p => {
    p.x += p.speedX;
    p.y += p.speedY;

    if (p.x > 140) p.x = -140;
    if (p.x < -140) p.x = 140;
    if (p.y > 140) p.y = -140;
    if (p.y < -140) p.y = 140;

    const px = cx + p.x;
    const py = cy + p.y;

    ctx.beginPath();
    ctx.arc(px, py, p.radius, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(158, 172, 149, ${p.opacity * 0.6})`;
    ctx.fill();
  });
}

function drawScanningWave(cx, cy, r) {
  const waveProgress = (stateTime % 1.6) / 1.6;
  const currentR = r * (0.8 + waveProgress * 0.85);
  const alpha = (1 - waveProgress) * 0.6;

  ctx.beginPath();
  ctx.arc(cx, cy, currentR, 0, Math.PI * 2);
  ctx.strokeStyle = `rgba(98, 123, 82, ${alpha})`;
  ctx.lineWidth = 2;
  ctx.setLineDash([]);
  ctx.stroke();
}

function renderStaticOrb() {
  const centerX = width / 2;
  const centerY = height / 2;
  const baseRadius = Math.min(width, height) * 0.28;

  ctx.clearRect(0, 0, width, height);
  drawAmbientGlow(centerX, centerY, baseRadius);
  drawOrbitalRings(centerX, centerY, baseRadius, true);
  drawCoreSphere(centerX, centerY, baseRadius);
  drawConstellation(centerX, centerY, baseRadius);
  drawOrbitalRings(centerX, centerY, baseRadius, false);
}
