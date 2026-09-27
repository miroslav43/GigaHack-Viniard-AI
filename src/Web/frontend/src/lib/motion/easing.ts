// Easing curves shared by the app's animations (t and the result in [0, 1]).
export const easeOutCubic = (t: number) => 1 - (1 - t) ** 3;
export const easeInOutQuad = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
