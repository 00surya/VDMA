import { el, svg } from "./ui.mjs";
import { COLORS, percentageShares } from "./data.mjs";
const INK = "#777780";

function frame(title, height = 290, width = 900) {
  const chart = svg("svg", {
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": title,
    class: "chart-svg",
  });
  chart.append(svg("title", {}, title));
  return chart;
}
function axes(chart, max, height, width, left = 40) {
  for (let i = 0; i <= 4; i++) {
    const y = 22 + ((height - 62) * i) / 4;
    chart.append(
      svg("line", {
        x1: left,
        y1: y,
        x2: width - 18,
        y2: y,
        stroke: "#2d2d34",
        "stroke-dasharray": "4 6",
      }),
      svg(
        "text",
        {
          x: left - 12,
          y: y + 4,
          "text-anchor": "end",
          fill: INK,
          "font-size": 12,
        },
        Math.round(max * (1 - i / 4)),
      ),
    );
  }
}
export function lineChart(target, rows) {
  const width = Math.max(320, target.clientWidth),
    height = Math.min(330, width * 0.6),
    max = Math.max(
      4,
      Math.ceil(Math.max(0, ...rows.map((row) => row.value)) / 4) * 4,
    );
  const chart = frame(
    "Incident counts over the selected period",
    height,
    width,
  );
  axes(chart, max, height, width);
  const points = rows.map((row, index) => [
    40 + (index * (width - 58)) / Math.max(1, rows.length - 1),
    22 + (height - 62) * (1 - row.value / max),
  ]);
  if (points.length) {
    let path = `M${points[0].join(",")}`;
    for (let i = 1; i < points.length; i++) {
      const [x, y] = points[i],
        [px, py] = points[i - 1],
        mid = (px + x) / 2;
      path += ` C${mid},${py} ${mid},${y} ${x},${y}`;
    }
    chart.append(
      svg("path", {
        d: `${path} L${points.at(-1)[0]},${height - 40} L40,${height - 40} Z`,
        fill: "#ef44440b",
      }),
      svg("path", {
        d: path,
        fill: "none",
        stroke: "#f0444f",
        "stroke-width": 3,
      }),
    );
    rows.forEach((row, index) => {
      const [x, y] = points[index];
      const dot = svg("circle", {
        cx: x,
        cy: y,
        r: rows.length > 10 ? 3 : 5,
        fill: "#f0444f",
        tabindex: 0,
        "aria-label": `${row.label}: ${row.value} incidents`,
      });
      dot.append(svg("title", {}, `${row.label} · ${row.value} incidents`));
      chart.append(dot);
      if (rows.length <= 10 || index % 5 === 0 || index === rows.length - 1)
        chart.append(
          svg(
            "text",
            {
              x,
              y: height - 12,
              "text-anchor": "middle",
              fill: INK,
              "font-size": 12,
            },
            rows.length > 5 && width < 500
              ? row.label.replace("Sector ", "S")
              : row.label,
          ),
        );
    });
  }
  target.replaceChildren(chart);
}
export function barChart(target, rows, title = "Incidents by sector") {
  const width = Math.max(320, target.clientWidth),
    height = Math.min(290, width * 0.62),
    chart = frame(title, height, width),
    max = Math.max(
      4,
      Math.ceil(Math.max(0, ...rows.map((row) => row.value)) / 4) * 4,
    );
  axes(chart, max, height, width);
  const slot = (width - 58) / Math.max(1, rows.length),
    barWidth = Math.min(74, slot * 0.6);
  rows.forEach((row, index) => {
    const h = ((height - 62) * row.value) / max,
      x = 40 + slot * (index + 0.5) - barWidth / 2;
    const bar = svg("rect", {
      x,
      y: height - 40 - h,
      width: barWidth,
      height: h,
      rx: 4,
      fill: "#f59e0b",
      tabindex: 0,
      "aria-label": `${row.label}: ${row.value}`,
    });
    bar.append(svg("title", {}, `${row.label} · ${row.value} incidents`));
    chart.append(
      bar,
      svg(
        "text",
        {
          x: x + barWidth / 2,
          y: height - 13,
          "text-anchor": "middle",
          fill: INK,
          "font-size": 12,
        },
        row.shortLabel
          ? row.shortLabel.length > Math.max(3, Math.floor(slot / 7))
            ? `${row.shortLabel.slice(0, Math.max(2, Math.floor(slot / 7) - 1))}…`
            : row.shortLabel
          : rows.length > 5 && width < 500
            ? row.label.replace("Sector ", "S")
            : row.label,
      ),
    );
  });
  target.replaceChildren(chart);
}
export function donutChart(target, rows) {
  const container = el("div", null, "donut-layout"),
    chart = svg("svg", {
      viewBox: "0 0 240 240",
      role: "img",
      "aria-label": "Incident type distribution",
      class: "donut-chart",
    });
  const total = rows.reduce((sum, row) => sum + row.value, 0),
    length = 2 * Math.PI * 84;
  chart.append(
    svg("circle", {
      cx: 120,
      cy: 120,
      r: 84,
      fill: "none",
      stroke: "#29292f",
      "stroke-width": 30,
    }),
  );
  let offset = 0;
  const shares = percentageShares(rows.map((row) => row.value));
  const legend = el("div", null, "chart-legend");
  rows.forEach((row, index) => {
    const segment = total ? (row.value / total) * length : 0,
      color = COLORS[index % COLORS.length];
    const circle = svg("circle", {
      cx: 120,
      cy: 120,
      r: 84,
      fill: "none",
      stroke: color,
      "stroke-width": 30,
      "stroke-dasharray": `${segment} ${length - segment}`,
      "stroke-dashoffset": -offset,
      transform: "rotate(-90 120 120)",
      tabindex: 0,
      "aria-label": `${row.label}: ${row.value}`,
    });
    circle.append(svg("title", {}, `${row.label} · ${row.value}`));
    chart.append(circle);
    offset += segment;
    const entry = el("div"),
      dot = el("i");
    dot.style.background = color;
    entry.append(dot, el("span", row.label), el("strong", `${shares[index]}%`));
    legend.append(entry);
  });
  chart.append(
    svg(
      "text",
      {
        x: 120,
        y: 118,
        "text-anchor": "middle",
        fill: "#eeeef2",
        "font-size": 32,
        "font-weight": 600,
      },
      total,
    ),
    svg(
      "text",
      { x: 120, y: 140, "text-anchor": "middle", fill: INK, "font-size": 12 },
      "INCIDENTS",
    ),
  );
  if (!total)
    legend.append(el("p", "No observations for this period.", "muted"));
  container.append(chart, legend);
  target.replaceChildren(container);
}
