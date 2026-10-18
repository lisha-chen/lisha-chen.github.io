(function () {
    var hoveredAreas = new Set();
    var resizeObserver = new ResizeObserver(updateHoveredAreas);

    function updateHoveredAreas() {
        hoveredAreas.forEach(function (area) {
            var style = getComputedStyle(area);
            var isPage = area === document.scrollingElement;
            var visible = style.scrollbarWidth !== 'none';
            var canScrollX = isPage
                ? !/^(hidden|clip)$/.test(style.overflowX)
                : /^(auto|scroll|overlay)$/.test(style.overflowX);
            var canScrollY = isPage
                ? !/^(hidden|clip)$/.test(style.overflowY)
                : /^(auto|scroll|overlay)$/.test(style.overflowY);

            area.classList.toggle('scrollbar-hover-x', visible && canScrollX &&
                area.clientWidth > 0 && area.scrollWidth > area.clientWidth);
            area.classList.toggle('scrollbar-hover-y', visible && canScrollY &&
                area.clientHeight > 0 && area.scrollHeight > area.clientHeight);
        });
    }

    // Recheck hovered areas when their viewport or content changes size.
    function observeHoveredAreas() {
        resizeObserver.disconnect();
        hoveredAreas.forEach(function (area) {
            resizeObserver.observe(area);
            Array.from(area.children).forEach(function (child) {
                resizeObserver.observe(child);
            });
        });
        updateHoveredAreas();
    }

    function clearArea(area) {
        area.classList.remove('scrollbar-hover-x', 'scrollbar-hover-y');
    }

    function clearHoveredAreas() {
        hoveredAreas.forEach(clearArea);
        hoveredAreas.clear();
        resizeObserver.disconnect();
    }

    document.addEventListener('pointerover', function (event) {
        if (event.pointerType === 'touch') {
            clearHoveredAreas();
            return;
        }

        var nextAreas = new Set();
        event.composedPath().forEach(function (element) {
            if (!(element instanceof Element)) {
                return;
            }

            var style = getComputedStyle(element);
            if (element === document.scrollingElement ||
                /^(auto|scroll|overlay)$/.test(style.overflowX) ||
                /^(auto|scroll|overlay)$/.test(style.overflowY)) {
                nextAreas.add(element);
            }
        });

        hoveredAreas.forEach(function (area) {
            if (!nextAreas.has(area)) {
                clearArea(area);
            }
        });
        hoveredAreas = nextAreas;
        observeHoveredAreas();
    });

    document.addEventListener('pointerout', function (event) {
        hoveredAreas.forEach(function (area) {
            if (!event.relatedTarget || !area.contains(event.relatedTarget)) {
                clearArea(area);
                hoveredAreas.delete(area);
            }
        });
        observeHoveredAreas();
    });
    document.addEventListener('pointercancel', clearHoveredAreas);
    window.addEventListener('blur', clearHoveredAreas);
}());
