"""Real browser regression for app/system theme independence; synthetic UI only."""
import json
import re
import pytest
from playwright.sync_api import expect


def luminance(rgb):
    channels = [int(n) / 255 for n in re.findall(r"\d+", rgb)[:3]]
    linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in channels]
    return sum(c * w for c, w in zip(linear, [.2126, .7152, .0722]))


@pytest.mark.parametrize("system_theme", ["light", "dark"])
@pytest.mark.parametrize("app_theme", ["light", "dark"])
def test_dialog_theme_contrast(browser, app_url, tmp_path, system_theme, app_theme):
    context = browser.new_context(color_scheme=system_theme, viewport={"width":1280, "height":900})
    try:
        page = context.new_page()
        page.goto(app_url)
        page.get_by_role("combobox", name="Theme", exact=True).select_option(app_theme)
        page.get_by_role("button", name="New project", exact=True).first.click()
        dialog = page.get_by_role("dialog")
        expect(dialog).to_be_visible()
        styles = dialog.evaluate("""el => {
          const s=getComputedStyle(el), i=getComputedStyle(el.querySelector('input'));
          return {background:s.backgroundColor, color:s.color, colorScheme:s.colorScheme,
                  inputBackground:i.backgroundColor, inputColor:i.color};
        }""")
        expected = "rgb(255, 255, 255)" if app_theme == "light" else "rgb(23, 29, 25)"
        assert styles["background"] == expected
        assert styles["colorScheme"] == app_theme
        a, b = sorted([luminance(styles["background"]), luminance(styles["color"])])
        ratio = (b + .05) / (a + .05)
        assert ratio >= 4.5, styles
        a, b = sorted([luminance(styles["inputBackground"]), luminance(styles["inputColor"])])
        input_ratio = (b + .05) / (a + .05)
        assert input_ratio >= 4.5, styles
        page.screenshot(path=str(tmp_path / "dialog.png"), full_page=True)
        (tmp_path / "contrast.json").write_text(json.dumps({"system_theme":system_theme,
            "app_theme":app_theme, "styles":styles, "contrast_ratio":ratio,
            "input_contrast_ratio":input_ratio}, indent=2))
        dialog.get_by_role("button", name="Close", exact=True).click()
        page.reload()
        expect(page.get_by_role("combobox", name="Theme", exact=True)).to_have_value(app_theme)
    finally:
        context.close()
