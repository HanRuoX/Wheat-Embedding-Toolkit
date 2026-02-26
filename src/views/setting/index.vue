<template>
  <div class="setting-page">
    <div class="setting-layout">
      <section class="setting-header">
        <h1 class="setting-title">{{ $t("setting.title") }}</h1>
        <div class="setting-chip-row">
          <span class="setting-chip">
            {{ $t("setting.language") }}: {{ currentLanguageLabel }}
          </span>
          <span class="setting-chip">
            {{ $t("setting.theme") }}: {{ currentThemeLabel }}
          </span>
        </div>
      </section>

      <div class="setting-grid">
        <a-card class="setting-card" :bordered="false">
          <div class="setting-item">
            <div class="setting-label">{{ $t("setting.language") }}</div>
            <a-radio-group
              class="setting-control"
              v-model:value="selectedLanguage"
              @change="changeLanguage"
            >
              <a-radio-button value="zh">{{ $t("setting.languageZh") }}</a-radio-button>
              <a-radio-button value="en">{{ $t("setting.languageEn") }}</a-radio-button>
            </a-radio-group>
          </div>
        </a-card>

        <a-card class="setting-card" :bordered="false">
          <div class="setting-item">
            <div class="setting-label">{{ $t("setting.theme") }}</div>
            <a-radio-group
              class="setting-control theme-mode-group"
              v-model:value="selectedTheme"
              @change="changeTheme"
            >
              <a-radio-button value="light">{{ $t("setting.themeLight") }}</a-radio-button>
              <a-radio-button value="dark">{{ $t("setting.themeDark") }}</a-radio-button>
              <a-radio-button value="system">{{ $t("setting.themeSystem") }}</a-radio-button>
            </a-radio-group>
          </div>
        </a-card>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import i18n from "@/locales/i18n";
import { computed, ref, watch } from "vue";
import { usePreferenceStore, type AppTheme } from "@/stores/Preference";

const preferenceStore = usePreferenceStore();
const selectedLanguage = ref(preferenceStore.language);
const selectedTheme = ref<AppTheme>(preferenceStore.themeMode);

const currentLanguageLabel = computed(() =>
  preferenceStore.language === "zh"
    ? i18n.global.t("setting.languageZh")
    : i18n.global.t("setting.languageEn")
);
const currentThemeLabel = computed(() => {
  if (preferenceStore.themeMode === "system") {
    const currentResolvedTheme =
      preferenceStore.resolvedTheme === "dark"
        ? i18n.global.t("setting.themeDark")
        : i18n.global.t("setting.themeLight");
    return `${i18n.global.t("setting.themeSystem")} (${currentResolvedTheme})`;
  }
  return preferenceStore.themeMode === "dark"
    ? i18n.global.t("setting.themeDark")
    : i18n.global.t("setting.themeLight");
});

const changeLanguage = () => {
  preferenceStore.setLanguage(selectedLanguage.value);
  i18n.global.locale.value = selectedLanguage.value;
  location.reload();
};

const changeTheme = () => {
  preferenceStore.setThemeMode(selectedTheme.value);
};

watch(
  () => preferenceStore.themeMode,
  (themeMode) => {
    selectedTheme.value = themeMode;
  }
);
</script>

<style scoped>
.setting-page {
  height: 100%;
  min-height: 0;
  overflow: auto;
  padding: 24px;
  box-sizing: border-box;
}

.setting-layout {
  max-width: 980px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.setting-header {
  border-radius: 14px;
  border: 1px solid var(--panel-border);
  background: linear-gradient(
    135deg,
    color-mix(in srgb, var(--panel-bg-strong) 86%, var(--accent) 14%),
    var(--panel-bg-strong)
  );
  padding: 18px 20px;
}

.setting-title {
  margin: 0;
  font-size: 22px;
  line-height: 1.2;
  color: var(--text-primary);
}

.setting-chip-row {
  margin-top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.setting-chip {
  display: inline-flex;
  align-items: center;
  height: 28px;
  padding: 0 10px;
  border-radius: 999px;
  border: 1px solid var(--panel-border);
  background: color-mix(in srgb, var(--panel-bg) 88%, var(--accent) 12%);
  color: var(--text-secondary);
  font-size: 12px;
}

.setting-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(280px, 1fr));
  gap: 16px;
}

.setting-card {
  border-radius: 14px;
  border: 1px solid var(--panel-border);
  background: var(--panel-bg);
}

.setting-card :deep(.ant-card-body) {
  padding: 18px 20px;
}

.setting-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 64px;
}

.setting-label {
  font-size: 16px;
  color: var(--text-primary);
  font-weight: 600;
}

.setting-control {
  flex-shrink: 0;
}

.theme-mode-group {
  display: flex;
  flex-wrap: wrap;
}

@media (max-width: 900px) {
  .setting-page {
    padding: 16px;
  }

  .setting-grid {
    grid-template-columns: 1fr;
  }

  .setting-item {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
