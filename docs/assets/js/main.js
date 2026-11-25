(() => {
  "use strict";

  const root = document.documentElement;
  const title = document.querySelector("title");
  const themeColor = document.querySelector('meta[name="theme-color"]');
  const systemTheme = window.matchMedia?.("(prefers-color-scheme: dark)");
  const languageKey = "pram-lang";
  const themeKey = "pram-theme";

  const messages = {
    en: {
      useDark: "Switch to dark theme",
      useLight: "Switch to light theme",
      openMenu: "Open navigation",
      closeMenu: "Close navigation",
      copy: "Copy BibTeX",
      copied: "Copied",
      manual: "Use your device’s Copy command to copy the selected citation."
    },
    zh: {
      useDark: "切换至深色主题",
      useLight: "切换至浅色主题",
      openMenu: "展开导航",
      closeMenu: "收起导航",
      copy: "复制 BibTeX",
      copied: "已复制",
      manual: "请使用设备的复制命令复制已选中的引用内容。"
    }
  };

  function readPreference(key) {
    try {
      return window.localStorage.getItem(key);
    } catch {
      return null;
    }
  }

  function savePreference(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch {
      // The document retains the current selection when storage is unavailable.
    }
  }

  function validTheme(value) {
    return value === "light" || value === "dark" ? value : null;
  }

  let preferredTheme = validTheme(readPreference(themeKey));

  function applyLanguage(value) {
    const language = value === "zh" ? "zh" : "en";
    root.dataset.lang = language;
    root.lang = language === "zh" ? "zh-CN" : "en";
    document.title = title.dataset[language];
  }

  function applyTheme() {
    const theme = preferredTheme ?? (systemTheme?.matches ? "dark" : "light");
    root.dataset.theme = theme;
    themeColor.content = theme === "dark" ? "#0e0f13" : "#f6f4ef";
  }

  root.classList.add("js");
  applyLanguage(readPreference(languageKey));
  applyTheme();

  function makeScenarioFilter(widget) {
    const card = widget.querySelector("#sensing-card");
    const table = card.querySelector("table");
    const scroller = card.querySelector(".table-scroll");
    const groupedElements = table.querySelectorAll("[data-scenario]");
    const columnScenarios = Array.from(table.tHead.rows[1].cells, cell => cell.dataset.scenario);

    return (filter, panel) => {
      for (const element of groupedElements) {
        element.hidden = filter !== "all" && element.dataset.scenario !== filter;
      }

      for (const row of table.tBodies[0].rows) {
        columnScenarios.forEach((scenario, index) => {
          row.cells[index + 1].hidden = filter !== "all" && scenario !== filter;
        });
      }

      if (card.parentElement !== panel) {
        panel.appendChild(card);
      }
      scroller.scrollLeft = 0;
    };
  }

  function initializeTabs(widget, onLayoutChange) {
    const tabs = Array.from(widget.querySelectorAll('[role="tab"]'));
    const panels = tabs.map(tab => document.getElementById(tab.getAttribute("aria-controls")));
    const filterScenario = widget.dataset.tabs === "sensing" ? makeScenarioFilter(widget) : null;

    function activate(index, moveFocus) {
      tabs.forEach((tab, position) => {
        const selected = position === index;
        tab.setAttribute("aria-selected", String(selected));
        tab.tabIndex = selected ? 0 : -1;
        panels[position].hidden = !selected;
      });

      if (filterScenario) {
        filterScenario(tabs[index].dataset.filter, panels[index]);
      }
      if (moveFocus) {
        tabs[index].focus();
      }
      onLayoutChange();
    }

    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => activate(index, false));
      tab.addEventListener("keydown", event => {
        let next;
        switch (event.key) {
          case "ArrowRight":
            next = (index + 1) % tabs.length;
            break;
          case "ArrowLeft":
            next = (index + tabs.length - 1) % tabs.length;
            break;
          case "Home":
            next = 0;
            break;
          case "End":
            next = tabs.length - 1;
            break;
          default:
            return;
        }
        event.preventDefault();
        activate(next, true);
      });
    });
  }

  function initializeSectionNavigation(header, navigation) {
    const sections = Array.from(document.querySelectorAll("[data-nav-section]"));
    const links = Array.from(navigation.querySelectorAll('a[href^="#"]'));
    let currentSection = "";
    let pendingFrame = 0;

    function update() {
      pendingFrame = 0;
      const readingLine = header.getBoundingClientRect().bottom + 24;
      let active = sections[0].dataset.navSection;

      for (const section of sections) {
        if (section.getBoundingClientRect().top > readingLine) break;
        active = section.dataset.navSection;
      }

      if (Math.ceil(window.scrollY + root.clientHeight) >= root.scrollHeight - 2) {
        active = sections[sections.length - 1].dataset.navSection;
      }
      if (active === currentSection) return;

      currentSection = active;
      for (const link of links) {
        if (link.hash === `#${active}`) {
          link.setAttribute("aria-current", "location");
        } else {
          link.removeAttribute("aria-current");
        }
      }
    }

    function schedule() {
      if (!pendingFrame) pendingFrame = window.requestAnimationFrame(update);
    }

    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver(schedule, {
        rootMargin: "-64px 0px 0px 0px",
        threshold: [0, 1]
      });
      sections.forEach(section => observer.observe(section));
    }

    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule, { passive: true });
    window.addEventListener("load", schedule, { once: true });
    update();
    return schedule;
  }

  function copyWithSelection(text) {
    const previousFocus = document.activeElement;
    const buffer = document.createElement("textarea");
    buffer.className = "clipboard-buffer";
    buffer.value = text;
    buffer.readOnly = true;
    buffer.tabIndex = -1;
    document.body.appendChild(buffer);

    try {
      buffer.focus({ preventScroll: true });
      buffer.select();
      buffer.setSelectionRange(0, buffer.value.length);
      return document.execCommand("copy");
    } catch {
      return false;
    } finally {
      buffer.remove();
      if (previousFocus instanceof HTMLElement) {
        previousFocus.focus({ preventScroll: true });
      }
    }
  }

  function initialize() {
    const header = document.getElementById("site-header");
    const navigation = document.getElementById("section-nav");
    const menuButton = document.getElementById("menu-toggle");
    const themeButton = document.getElementById("theme-toggle");
    const languageButtons = document.querySelectorAll("[data-language]");
    const labeledElements = document.querySelectorAll("[data-label-en]");
    const figures = document.querySelectorAll("img[data-alt-en]");
    const copyButton = document.getElementById("copy-bibtex");
    const copyHelp = document.getElementById("copy-help");
    const copyStatus = document.getElementById("copy-status");
    const citation = document.getElementById("bibtex");
    const mobileLayout = window.matchMedia?.("(max-width: 60rem)");
    const scheduleNavigation = initializeSectionNavigation(header, navigation);
    let copyTimer = 0;
    let copying = false;

    function setButtonLabel(button, label) {
      button.setAttribute("aria-label", label);
      button.title = label;
    }

    function refreshThemeLabel() {
      const text = messages[root.dataset.lang];
      setButtonLabel(themeButton, root.dataset.theme === "dark" ? text.useLight : text.useDark);
    }

    function refreshMenuLabel() {
      const text = messages[root.dataset.lang];
      setButtonLabel(menuButton, header.dataset.open === "true" ? text.closeMenu : text.openMenu);
    }

    function refreshCopyLabel() {
      const text = messages[root.dataset.lang];
      const state = copyButton.dataset.state;
      setButtonLabel(copyButton, state === "copied" ? text.copied : text.copy);
      copyHelp.hidden = state !== "manual";
      copyStatus.textContent = state === "idle" ? "" : text[state];
    }

    function refreshLanguage() {
      const language = root.dataset.lang;
      for (const element of labeledElements) {
        element.setAttribute("aria-label", element.getAttribute(`data-label-${language}`));
      }
      for (const image of figures) {
        image.alt = image.getAttribute(`data-alt-${language}`);
      }
      for (const button of languageButtons) {
        button.setAttribute("aria-pressed", String(button.dataset.language === language));
      }
      refreshThemeLabel();
      refreshMenuLabel();
      refreshCopyLabel();
      scheduleNavigation();
    }

    function setMenu(open) {
      header.dataset.open = String(open);
      menuButton.setAttribute("aria-expanded", String(open));
      refreshMenuLabel();
    }

    for (const button of languageButtons) {
      button.addEventListener("click", () => {
        const next = button.dataset.language;
        applyLanguage(next);
        savePreference(languageKey, next);
        refreshLanguage();
      });
    }

    themeButton.addEventListener("click", () => {
      preferredTheme = root.dataset.theme === "dark" ? "light" : "dark";
      savePreference(themeKey, preferredTheme);
      applyTheme();
      refreshThemeLabel();
    });

    function followSystemTheme() {
      if (preferredTheme !== null) return;
      applyTheme();
      refreshThemeLabel();
    }

    if (systemTheme?.addEventListener) {
      systemTheme.addEventListener("change", followSystemTheme);
    } else {
      systemTheme?.addListener(followSystemTheme);
    }

    window.addEventListener("storage", event => {
      if (event.key === languageKey || event.key === null) {
        applyLanguage(readPreference(languageKey));
        refreshLanguage();
      }
      if (event.key === themeKey || event.key === null) {
        preferredTheme = validTheme(readPreference(themeKey));
        applyTheme();
        refreshThemeLabel();
      }
    });

    menuButton.addEventListener("click", () => {
      const open = header.dataset.open !== "true";
      setMenu(open);
      if (open) {
        const firstLink = navigation.querySelector('[aria-current="location"]') || navigation.querySelector("a");
        firstLink.focus({ preventScroll: true });
      }
    });

    header.addEventListener("click", event => {
      if (event.target.closest("a")) setMenu(false);
    });
    document.addEventListener("pointerdown", event => {
      if (header.dataset.open === "true" && !header.contains(event.target)) setMenu(false);
    });
    header.addEventListener("focusout", event => {
      if (event.relatedTarget && !header.contains(event.relatedTarget)) setMenu(false);
    });
    document.addEventListener("keydown", event => {
      if (event.key !== "Escape" || header.dataset.open !== "true") return;
      setMenu(false);
      menuButton.focus({ preventScroll: true });
    });
    window.addEventListener("resize", () => {
      if (mobileLayout && !mobileLayout.matches && header.dataset.open === "true") setMenu(false);
    }, { passive: true });

    document.querySelectorAll("[data-tabs]").forEach(widget => {
      initializeTabs(widget, scheduleNavigation);
    });

    copyButton.addEventListener("click", async () => {
      if (copying) return;
      copying = true;
      window.clearTimeout(copyTimer);
      const text = citation.textContent;
      let copied = false;

      try {
        if (navigator.clipboard?.writeText) {
          await navigator.clipboard.writeText(text);
          copied = true;
        }
      } catch {
        copied = false;
      }

      if (!copied) copied = copyWithSelection(text);
      if (!copied) {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(citation);
        selection?.removeAllRanges();
        selection?.addRange(range);
      }

      copyButton.dataset.state = copied ? "copied" : "manual";
      refreshCopyLabel();
      copying = false;

      if (copied) {
        copyTimer = window.setTimeout(() => {
          copyButton.dataset.state = "idle";
          refreshCopyLabel();
        }, 2400);
      }
    });

    refreshLanguage();
  }

  let mathRendered = false;

  function renderMath() {
    if (mathRendered || typeof window.renderMathInElement !== "function") return;
    const article = document.querySelector(".article");
    if (!article) return;
    mathRendered = true;
    window.renderMathInElement(article, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false }
      ],
      throwOnError: false,
      trust: false,
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"]
    });
    article.querySelectorAll(".katex-display").forEach(display => {
      display.tabIndex = 0;
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("readystatechange", initialize, { once: true });
  } else {
    initialize();
  }

  if (document.readyState === "complete") {
    renderMath();
  } else {
    document.addEventListener("DOMContentLoaded", renderMath, { once: true });
    window.addEventListener("load", renderMath, { once: true });
  }
})();
