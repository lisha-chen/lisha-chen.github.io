(function () {
    var button = document.getElementById('navToggle');
    var mobileLayout = window.matchMedia('(max-width: 1200px)');

    function setMenuOpen(open) {
        open = open && mobileLayout.matches;
        document.body.classList.toggle('nav-open', open);
        button.setAttribute('aria-expanded', String(open));
        button.setAttribute('aria-label', open ? 'Close navigation menu' : 'Open navigation menu');
    }

    button.addEventListener('click', function () {
        setMenuOpen(button.getAttribute('aria-expanded') !== 'true');
    });

    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape' && button.getAttribute('aria-expanded') === 'true') {
            setMenuOpen(false);
            button.focus();
        }
    });

    mobileLayout.addEventListener('change', function () {
        setMenuOpen(false);
    });
    setMenuOpen(false);
}());

(function () {
    var iconLink = document.querySelector('#header-icon-container a');
    var icon = iconLink && iconLink.querySelector('img');
    if (!icon) {
        return;
    }

    var originalSource = icon.getAttribute('src');
    var smileImage = new Image();
    smileImage.src = 'figures/photos/chibi/lisha-icon-smile-256.png';

    function updateIcon() {
        icon.src = iconLink.matches(':hover') || document.activeElement === iconLink
            ? smileImage.src
            : originalSource;
    }

    iconLink.addEventListener('mouseenter', updateIcon);
    iconLink.addEventListener('mouseleave', updateIcon);
    iconLink.addEventListener('focus', updateIcon);
    iconLink.addEventListener('blur', updateIcon);
}());
