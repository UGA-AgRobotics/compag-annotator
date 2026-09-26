try { document.documentElement.dataset.theme = localStorage.getItem("compag-theme") === "dark" ? "dark" : "light"; }
catch (_) { document.documentElement.dataset.theme = "light"; }
