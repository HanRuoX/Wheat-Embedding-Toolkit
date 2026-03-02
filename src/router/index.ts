import { createRouter, createWebHistory } from "vue-router";
import Layout from "@/views/layout/index.vue";
import i18n from "@/locales/i18n";
import bleIcon from "@/assets/ble-icon.png";
import serialIcon from "@/assets/serial-icon.png";
import audioIcon from "@/assets/audio-icon.png";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      component: Layout,
      children: [
        {
          path: "/",
          name: "tools",
          meta: {
            icon: "🛠️",
            title: i18n.global.t("menu.toolList"),
          },
          children: [
            {
              path: "/",
              name: "home",
              component: () => import("@/views/home/index.vue"),
              meta: {
                icon: "🐯",
                title: i18n.global.t("menu.home"),
                display: false,
              },
            },
            {
              path: "/tools/basic",
              name: "basic",
              component: () => import("@/views/tools/basic/index.vue"),
              meta: {
                icon: "🐼",
                title: i18n.global.t("menu.general"),
                display: true,
              },
            },
            {
              path: "/tools/flash",
              name: "flash",
              component: () => import("@/views/tools/flash/index.vue"),
              meta: {
                icon: "🐶",
                title: i18n.global.t("menu.flash"),
                display: true,
              },
            },
            {
              path: "/tools/partition",
              name: "partition",
              component: () => import("@/views/tools/partition/index.vue"),
              meta: {
                icon: "🐱",
                title: i18n.global.t("menu.partitionTable"),
                display: true,
              },
            },
            {
              path: "/tools/ble",
              name: "ble",
              component: () => import("@/views/tools/ble/index.vue"),
              meta: {
                icon: bleIcon,
                title: "BLE",
                display: true,
              },
            },
            {
              path: "/tools/serial",
              name: "serial",
              component: () => import("@/views/tools/serial/index.vue"),
              meta: {
                icon: serialIcon,
                title: i18n.global.t("menu.serialAssistant"),
                display: true,
              },
            },
            {
              path: "/tools/firmware",
              name: "firmware",
              component: () => import("@/views/tools/firmware/index.vue"),
              meta: {
                icon: "🐰",
                title: i18n.global.t("menu.firmware"),
                display: true,
              },
            },
            {
              path: "/tools/audio",
              name: "audio",
              component: () => import("@/views/tools/audio/index.vue"),
              meta: {
                icon: audioIcon,
                title: i18n.global.t("menu.audioTool"),
                display: true,
              },
            },
          ],
        },
        {
          path: "/setting",
          name: "setting",
          meta: {
            icon: "⚙️",
            title: i18n.global.t("menu.setting"),
            display: false,
          },
          component: () => import("@/views/setting/index.vue"),
        },
        {
          path: "/help",
          name: "help",
          meta: {
            icon: "📙",
            title: "帮助",
            display: false,
          },
          component: () => import("@/views/help/index.vue"),
        },
      ],
    },
  ],
});

export default router;
