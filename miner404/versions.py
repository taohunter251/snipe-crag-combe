from .pipeline import Draft, StagePlan, Step

ACTIVE_PLAN = StagePlan(
    drafts=(
        Draft(start='plain', model='gpt-6.1-sol', effort='high'),
        Draft(start='proportions', model='gpt-6.1-sol', effort='high'),
        Draft(start='plain', model='gpt-6.1-sol', effort='high'),
        Draft(start='proportions', model='gpt-6.1-sol', effort='high'),
        Draft(start='plain', model='gpt-6.1-sol', effort='high'),
        Draft(start='proportions', model='gpt-6.1-sol', effort='high'),
    ),
    steps=(
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), focus=('materials', 'shape'), fallback=('gpt-6.1-sol', 'high'), info=('gate',)),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), focus=('materials', 'shape'), fallback=('gpt-6.1-sol', 'high'), info=('gate',)),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high'), info=('gate',)),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high'), info=('gate',)),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high'), info=('gate',)),
        Step(rewrites=(('gpt-6.1-sol', 'max'),), fallback=('gpt-6.1-sol', 'xhigh'), info=('gate',), nudge=' Before anything else, compare the reference photo with the current model for small visible details that the model leaves out or simplifies, and add or sharpen the most visible of them, keeping the overall shape, proportions, pose and colours as they are.'),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high'), info=('gate',), nudge=' Before anything else, compare the reference photo with the current model for small visible details that the model leaves out or simplifies, and add or sharpen the most visible of them, keeping the overall shape, proportions, pose and colours as they are.'),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high'), info=('gate',), nudge=' Before anything else, compare the reference photo with the current model for small visible details that the model leaves out or simplifies, and add or sharpen the most visible of them, keeping the overall shape, proportions, pose and colours as they are.'),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high'), info=('gate',), nudge=" In this step, work on metal only. Compare each metal part in the renders with the same part in the photo. Bare metal (polished, brushed or plated) that looks gray, flat or like plastic in the renders gets a fully metallic finish with the photo's tint and polish, so it reads as metal; painted or coated metal stays paint. Leave every other part and every color as it is. Keep the shape, proportions, pose, parts and details as they are."),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high'), info=('gate',), nudge=' In this step, work on color only. A glossy or semi-gloss finish adds white highlights in the renders and washes out the color beneath, most of all for light warm colors. Give each painted, plastic or fabric part a color as rich as the photo shows and the matte or satin finish the photo shows, so the render keeps that richness; leave metal and parts that already match. Keep the shape, proportions, pose, parts and details as they are.'),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high'), info=('gate',), nudge=" In this step, work on color only. Compare each part of the renders with the same part in the photo: where a render looks paler, grayer or more washed out than the photo, which happens most with light warm colors and glossy paint, deepen that part's base color until it matches the photo. Leave parts that already match, and keep every hue. Keep the shape, proportions, pose, parts and details as they are."),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high'), patch=True, info=('gate',), nudge=" In this step, work on color only. Compare each part of the renders with the same part in the photo: where a render looks paler, grayer or more washed out than the photo, which happens most with light warm colors and glossy paint, deepen that part's base color until it matches the photo. Leave parts that already match, and keep every hue. Keep the shape, proportions, pose, parts and details as they are."),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high'), patch=True, info=('gate',), nudge=' Before anything else, compare the reference photo with the current model for small visible details that the model leaves out or simplifies, and add or sharpen the most visible of them, keeping the overall shape, proportions, pose and colours as they are.'),
    ),
    draft_keep=1,
    fill=0.999,
    render_cap=76,
    keep_dark=True,
    stand=True,
)
