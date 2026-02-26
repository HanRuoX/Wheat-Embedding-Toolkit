import { createApp } from "vue";
import router from "./router";
import App from "./App.vue";
import Antd from "ant-design-vue";
import useClipboard from "vue-clipboard3";
import i18n from "./locales/i18n";
import { createPinia } from "pinia";
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate'
const pinia = createPinia();
pinia.use(piniaPluginPersistedstate)

const userAgent = navigator.userAgent.toLowerCase();
if (userAgent.includes("mac")) {
  document.documentElement.setAttribute("data-os", "mac");
} else if (userAgent.includes("win")) {
  document.documentElement.setAttribute("data-os", "win");
} else {
  document.documentElement.setAttribute("data-os", "linux");
}

const { toClipboard } = useClipboard();
const app = createApp(App);

app.config.warnHandler = (msg, _instance, _trace) => {
  if (msg.includes("Slot \"default\" invoked outside of the render function")) return;
  console.warn(msg);
};
import "./assets/css/style.css";


app.directive("copy", (el, binding) => {
  el.addEventListener("click", () => {
    toClipboard(el.textContent);
  });
});

app.use(router).use(pinia).use(i18n).use(Antd).mount("#app");
