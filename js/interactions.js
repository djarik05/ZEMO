/**
 * ZEMO INTERACTION UTILITIES
 * Toast notification system, accessible modal controller,
 * and scroll-triggered reveal animations.
 */

let activeModal = null;

export function initInteractions() {
  initScrollReveals();
  initModalListeners();

  // Expose to window for global convenience
  window.zemo = window.zemo || {};
  window.zemo.showToast = showToast;
  window.zemo.showModal = showModal;
  window.zemo.closeModal = closeModal;
}

// 1. Toast Notification System
export function showToast(message, type = "info", duration = 3800) {
  let container = document.querySelector(".toast-container");
  if (!container) {
    container = document.createElement("div");
    container.className = "toast-container";
    document.body.appendChild(container);
  }

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const iconSvg = getToastIcon(type);
  toast.innerHTML = `
    ${iconSvg}
    <span class="toast-msg">${message}</span>
  `;

  container.appendChild(toast);

  // Trigger animation
  requestAnimationFrame(() => {
    toast.classList.add("show");
  });

  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => {
      toast.remove();
    }, 300);
  }, duration);
}

function getToastIcon(type) {
  if (type === "success") {
    return `<svg class="toast-icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/></svg>`;
  }
  if (type === "warning") {
    return `<svg class="toast-icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"/></svg>`;
  }
  return `<svg class="toast-icon" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/></svg>`;
}

// 2. Modal Controller
export function showModal(modalId) {
  const modal = document.getElementById(modalId);
  if (!modal) return;

  activeModal = modal;
  modal.classList.add("active");
  document.body.style.overflow = "hidden";

  // Focus on first action or close button
  const focusable = modal.querySelector("button, [href], input, select");
  focusable?.focus();
}

export function closeModal() {
  if (activeModal) {
    activeModal.classList.remove("active");
    activeModal = null;
    document.body.style.overflow = "";
  }
}

function initModalListeners() {
  document.querySelectorAll(".modal-overlay").forEach(overlay => {
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) {
        closeModal();
      }
    });
  });

  document.querySelectorAll("[data-close-modal]").forEach(btn => {
    btn.addEventListener("click", closeModal);
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && activeModal) {
      closeModal();
    }
  });
}

// 3. Scroll Reveal System
function initScrollReveals() {
  const reveals = document.querySelectorAll(".reveal, .reveal-fade");
  if (!reveals.length) return;

  if ("IntersectionObserver" in window) {
    const revealObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add("revealed");
          observer.unobserve(entry.target);
        }
      });
    }, {
      rootMargin: "0px 0px -40px 0px",
      threshold: 0.1
    });

    reveals.forEach(el => revealObserver.observe(el));
  } else {
    reveals.forEach(el => el.classList.add("revealed"));
  }
}
