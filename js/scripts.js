if (localStorage.getItem('theme') == 'dark') {
    darkMode();
} else {
    lightMode();
}

function switchButtons() {
    if (localStorage.getItem('theme') == 'dark') {
        document.getElementById('dark').setAttribute('style', 'display:none;');
        document.getElementById('light').setAttribute('style', '');
    } else {
        document.getElementById('light').setAttribute('style', 'display:none;');
        document.getElementById('dark').setAttribute('style', '');
    }
}

function darkMode() {
    document.getElementById('stylesheet').setAttribute('href', 'css/style_dark.css');
    document.getElementById('favicon').setAttribute('href', 'images/favicon_dark.ico');
    localStorage.setItem('theme', 'dark');
}

function lightMode() {
    document.getElementById('stylesheet').setAttribute('href', 'css/style.css');
    document.getElementById('favicon').setAttribute('href', 'images/favicon.ico');
    localStorage.setItem('theme', 'light');
}
