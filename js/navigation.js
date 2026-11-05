/**
 * ZEMO NAVIGATION CONTROLLER
 * Handles sticky navbar behavior, smooth anchor scrolling,
 * IntersectionObserver scrollspy, and mobile drawer accessibility.
 */

export function initNavigation() {
  const header = document.querySelector(".nav-header");
  const navLinks = document.querySelectorAll(".nav-link");
  const mobileToggle = document.querySelector(".mobile-toggle");
  const navMenu = document.querySelector(".nav-links");
  const startAnalysisBtn = document.getElementById("nav-start-analysis");

  // 1. Sticky Header Background Transition on Scroll
  window.addEventListener("scroll", () => {
    if (window.scrollY > 24) {
      header?.classList.add("scrolled");
    } else {
      header?.classList.remove("scrolled");
    }
  }, { passive: true });

  // 2. Smooth Scrolling to Sections
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener("click", function (e) {
      const targetId = this.getAttribute("href");
      if (!targetId || targetId === "#") return;

      const targetEl = document.querySelector(targetId);
      if (targetEl) {
        e.preventDefault();
        closeMobileMenu();

        const headerOffset = 76;
        const elementPosition = targetEl.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

        window.scrollTo({
          top: offsetPosition,
          behavior: "smooth"
        });
      }
    });
  });

  // 3. CTA Quick-Scroll
  startAnalysisBtn?.addEventListener("click", (e) => {
    e.preventDefault();
    const datasetSection = document.getElementById("dataset");
    if (datasetSection) {
      datasetSection.scrollIntoView({ behavior: "smooth" });
      const dropzone = document.querySelector(".upload-dropzone");
      if (dropzone) {
        setTimeout(() => {
          dropzone.classList.add("drag-active");
          setTimeout(() => dropzone.classList.remove("drag-active"), 800);
        }, 500);
      }
    }
  });

  // 4. Scrollspy via IntersectionObserver
  const sections = document.querySelectorAll("section[id]");
  if ("IntersectionObserver" in window && sections.length > 0) {
    const observerOptions = {
      root: null,
      rootMargin: "-25% 0px -65% 0px",
      threshold: 0
    };

    const sectionObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const currentId = entry.target.getAttribute("id");
          navLinks.forEach(link => {
            const href = link.getAttribute("href");
            if (href === `#${currentId}`) {
              link.classList.add("active");
            } else {
              link.classList.remove("active");
            }
          });
        }
      });
    }, observerOptions);

    sections.forEach(sec => sectionObserver.observe(sec));
  }

  // 5. Mobile Drawer Toggle
  if (mobileToggle && navMenu) {
    mobileToggle.addEventListener("click", () => {
      const isOpen = navMenu.classList.toggle("open");
      mobileToggle.setAttribute("aria-expanded", String(isOpen));
      document.body.style.overflow = isOpen ? "hidden" : "";
    });

    // Close on Escape Key
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && navMenu.classList.contains("open")) {
        closeMobileMenu();
        mobileToggle.focus();
      }
    });

    // Close on Window Resize
    window.addEventListener("resize", () => {
      if (window.innerWidth > 860 && navMenu.classList.contains("open")) {
        closeMobileMenu();
      }
    });
  }

  function closeMobileMenu() {
    if (navMenu?.classList.contains("open")) {
      navMenu.classList.remove("open");
      mobileToggle?.setAttribute("aria-expanded", "false");
      document.body.style.overflow = "";
    }
  }
}
