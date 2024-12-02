// script.js
const themeButton = document.getElementById('theme-button');
const darkTheme = 'dark-theme';

themeButton.addEventListener('click', () => {
    document.body.classList.toggle(darkTheme);
});
