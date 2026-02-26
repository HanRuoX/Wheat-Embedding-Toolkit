import { defineStore } from "pinia";

export type AppLanguage = "zh" | "en";
export type AppTheme = "dark" | "light" | "system";
export type ResolvedTheme = "dark" | "light";

const getInitialLanguage = (): AppLanguage => {
  const saved = localStorage.getItem("language");
  return saved === "en" ? "en" : "zh";
};

const getInitialTheme = (): AppTheme => {
  const saved = localStorage.getItem("themeMode");
  if (saved === "light" || saved === "dark" || saved === "system") {
    return saved;
  }
  return "dark";
};

const getInitialSystemTheme = (): ResolvedTheme => {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return "dark";
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
};

export const usePreferenceStore = defineStore("preference", {
  state: () => ({
    language: getInitialLanguage() as AppLanguage,
    themeMode: getInitialTheme() as AppTheme,
    systemTheme: getInitialSystemTheme() as ResolvedTheme,
  }),
  getters: {
    resolvedTheme: (state): ResolvedTheme =>
      state.themeMode === "system" ? state.systemTheme : state.themeMode,
  },
  actions: {
    setLanguage(language: AppLanguage) {
      this.language = language;
      localStorage.setItem("language", language);
    },
    setThemeMode(themeMode: AppTheme) {
      this.themeMode = themeMode;
      localStorage.setItem("themeMode", themeMode);
    },
    setSystemTheme(theme: ResolvedTheme) {
      this.systemTheme = theme;
    },
  },
  persist: true,
});
