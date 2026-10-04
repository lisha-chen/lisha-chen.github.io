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
