use crate::Neuron;
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;
use std::fs;
use std::path::Path;

const BG: RGBColor = RGBColor(8, 6, 24);
const PANEL_BG: RGBColor = RGBColor(20, 10, 46);
const GRID: RGBColor = RGBColor(92, 46, 160);
const TEXT: RGBColor = RGBColor(244, 239, 231);
const PURPLE: RGBColor = RGBColor(164, 82, 255);
const GREEN_NEON: RGBColor = RGBColor(181, 223, 0);
const ORANGE_FIRE: RGBColor = RGBColor(255, 120, 0);
const GOLD: RGBColor = RGBColor(255, 180, 50);

fn prepare_output_path(path: &str) {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).unwrap();
        }
    }
}

fn styled_root(path: &str, size: (u32, u32)) -> DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> {
    prepare_output_path(path);
    let root = BitMapBackend::new(path, size).into_drawing_area();
    root.fill(&BG).unwrap();
    root
}

fn paint_panel_background(area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>) {
    area.fill(&PANEL_BG).unwrap();
}

fn style_mesh<'a, DB: DrawingBackend>(
    chart: &mut ChartContext<'a, DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    x_desc: &str,
    y_desc: &str,
) {
    chart
        .configure_mesh()
        .bold_line_style(GRID.mix(0.45))
        .light_line_style(GRID.mix(0.18))
        .axis_style(TEXT.mix(0.75))
        .label_style(("sans-serif", 13).into_font().color(&TEXT))
        .x_desc(x_desc)
        .y_desc(y_desc)
        .draw()
        .unwrap();
}

fn style_legend<'a, DB: DrawingBackend + 'a>(
    chart: &mut ChartContext<'a, DB, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
) {
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(TEXT.mix(0.35))
        .background_style(BG.mix(0.9))
        .label_font(("sans-serif", 12).into_font().color(&TEXT))
        .draw()
        .unwrap();
}

fn draw_prediction_panel(
    area: &DrawingArea<BitMapBackend<'_>, plotters::coord::Shift>,
    dataset: &[(f64, f64)],
    neuron: &Neuron,
    title: &str,
    show_point_labels: bool,
) {
    paint_panel_background(area);

    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 16).into_font().color(&TEXT))
        .margin(28)
        .x_label_area_size(42)
        .y_label_area_size(52)
        .build_cartesian_2d(0f64..100f64, -10f64..110f64)
        .unwrap();

    style_mesh(&mut chart, "energia", "distância");

    let line: Vec<(f64, f64)> = (0..=100)
        .map(|x| (x as f64, neuron.predict(x as f64)))
        .collect();

    let legend_label = format!("previsão (w={:.2}, b={:.2})", neuron.weight, neuron.bias);
    chart
        .draw_series(LineSeries::new(line, PURPLE.stroke_width(4)))
        .unwrap()
        .label(legend_label)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], PURPLE.stroke_width(4)));

    for (energy, actual) in dataset {
        let predicted = neuron.predict(*energy);
        let error = predicted - actual;
        let mid_y = (predicted + actual) / 2.0;

        chart
            .draw_series(LineSeries::new(
                vec![(*energy, predicted), (*energy, *actual)],
                ORANGE_FIRE.stroke_width(3),
            ))
            .unwrap();

        chart
            .draw_series(std::iter::once(Circle::new(
                (*energy, predicted),
                4,
                PURPLE.mix(0.9).filled(),
            )))
            .unwrap();

        chart
            .draw_series(std::iter::once(Text::new(
                format!("{:.1}", error),
                (*energy + 1.8, mid_y),
                ("sans-serif", 10).into_font().color(&ORANGE_FIRE),
            )))
            .unwrap();
    }

    chart
        .draw_series(std::iter::once(Circle::new((0.0, -999.0), 0, TRANSPARENT)))
        .unwrap()
        .label("gabarito")
        .legend(|(x, y)| Circle::new((x + 10, y), 6, GREEN_NEON.filled()));

    chart
        .draw_series(std::iter::once(Circle::new((0.0, -999.0), 0, TRANSPARENT)))
        .unwrap()
        .label("erro")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 24, y)], ORANGE_FIRE.stroke_width(3)));

    chart
        .draw_series(
            dataset
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 6, GREEN_NEON.filled())),
        )
        .unwrap();

    if show_point_labels {
        chart
            .draw_series(dataset.iter().map(|(x, y)| {
                Text::new(
                    format!("({}, {})", x, y),
                    (*x + 2.0, *y + 3.0),
                    ("sans-serif", 11).into_font().color(&TEXT),
                )
            }))
            .unwrap();
    }

    style_legend(&mut chart);
}

pub fn plot_slopes(path: &str) {
    let root = styled_root(path, (700, 500));
    paint_panel_background(&root);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Efeito do weight (m) — mesmo bias, inclinações diferentes",
            ("sans-serif", 18).into_font().color(&TEXT),
        )
        .margin(28)
        .x_label_area_size(42)
        .y_label_area_size(52)
        .build_cartesian_2d(0f64..10f64, -5f64..25f64)
        .unwrap();

    style_mesh(&mut chart, "x (entrada)", "y (saída)");

    let lines = [
        (0.5, 0.0, "m=0.5, b=0", GOLD),
        (1.0, 0.0, "m=1.0, b=0", PURPLE),
        (2.0, 0.0, "m=2.0, b=0", GREEN_NEON),
    ];

    for (m, b, label, color) in lines {
        let points: Vec<(f64, f64)> = (0..=100)
            .map(|i| {
                let x = i as f64 * 0.1;
                (x, m * x + b)
            })
            .collect();

        chart
            .draw_series(LineSeries::new(points, color.stroke_width(4)))
            .unwrap()
            .label(label)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 24, y)], color.stroke_width(4))
            });
    }

    style_legend(&mut chart);
    root.present().unwrap();
    println!("Salvo em {path}");
}

pub fn plot_biases(path: &str) {
    let root = styled_root(path, (700, 500));
    paint_panel_background(&root);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Efeito do bias (b) — mesmo weight, deslocamentos diferentes",
            ("sans-serif", 18).into_font().color(&TEXT),
        )
        .margin(28)
        .x_label_area_size(42)
        .y_label_area_size(52)
        .build_cartesian_2d(0f64..10f64, -2f64..22f64)
        .unwrap();

    style_mesh(&mut chart, "x (entrada)", "y (saída)");

    let lines = [
        (1.0, 0.0, "m=1, b=0", GOLD),
        (1.0, 5.0, "m=1, b=5", PURPLE),
        (1.0, 10.0, "m=1, b=10", GREEN_NEON),
    ];

    for (m, b, label, color) in lines {
        let points: Vec<(f64, f64)> = (0..=100)
            .map(|i| {
                let x = i as f64 * 0.1;
                (x, m * x + b)
            })
            .collect();

        chart
            .draw_series(LineSeries::new(points, color.stroke_width(4)))
            .unwrap()
            .label(label)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 24, y)], color.stroke_width(4))
            });
    }

    style_legend(&mut chart);
    root.present().unwrap();
    println!("Salvo em {path}");
}

pub fn plot_error(dataset: &[(f64, f64)], neuron: &Neuron, path: &str) {
    let root = styled_root(path, (700, 500));
    let title = format!("Neurônio — weight={:.2}, bias={:.2}", neuron.weight, neuron.bias);
    draw_prediction_panel(&root, dataset, neuron, &title, true);
    root.present().unwrap();
    println!("Salvo em {path}");
}

pub fn plot_equation_comparison(path: &str) {
    let root = styled_root(path, (1400, 500));
    let (left, right) = root.split_horizontally(700);

    paint_panel_background(&left);
    paint_panel_background(&right);

    {
        let mut chart = ChartBuilder::on(&left)
            .caption(
                "Efeito do weight (m) — mesma origem, inclinações diferentes",
                ("sans-serif", 15).into_font().color(&TEXT),
            )
            .margin(28)
            .x_label_area_size(42)
            .y_label_area_size(52)
            .build_cartesian_2d(0f64..10f64, -5f64..25f64)
            .unwrap();

        style_mesh(&mut chart, "x", "y");

        for (m, b, label, color) in [
            (0.5, 0.0, "m=0.5, b=0", GOLD),
            (1.0, 0.0, "m=1.0, b=0", PURPLE),
            (2.0, 0.0, "m=2.0, b=0", GREEN_NEON),
        ] {
            let points: Vec<(f64, f64)> = (0..=100)
                .map(|i| {
                    let x = i as f64 * 0.1;
                    (x, m * x + b)
                })
                .collect();

            chart
                .draw_series(LineSeries::new(points, color.stroke_width(4)))
                .unwrap()
                .label(label)
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 24, y)], color.stroke_width(4))
                });
        }

        style_legend(&mut chart);
    }

    {
        let mut chart = ChartBuilder::on(&right)
            .caption(
                "Efeito do bias (b) — mesma inclinação, deslocamentos diferentes",
                ("sans-serif", 15).into_font().color(&TEXT),
            )
            .margin(28)
            .x_label_area_size(42)
            .y_label_area_size(52)
            .build_cartesian_2d(0f64..10f64, -2f64..22f64)
            .unwrap();

        style_mesh(&mut chart, "x", "y");

        for (m, b, label, color) in [
            (1.0, 0.0, "m=1, b=0", GOLD),
            (1.0, 5.0, "m=1, b=5", PURPLE),
            (1.0, 10.0, "m=1, b=10", GREEN_NEON),
        ] {
            let points: Vec<(f64, f64)> = (0..=100)
                .map(|i| {
                    let x = i as f64 * 0.1;
                    (x, m * x + b)
                })
                .collect();

            chart
                .draw_series(LineSeries::new(points, color.stroke_width(4)))
                .unwrap()
                .label(label)
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 24, y)], color.stroke_width(4))
                });
        }

        style_legend(&mut chart);
    }

    root.present().unwrap();
    println!("Salvo em {path}");
}

pub fn plot_comparison(dataset: &[(f64, f64)], before: &Neuron, after: &Neuron, path: &str) {
    let root = styled_root(path, (1400, 500));
    let (left, right) = root.split_horizontally(700);

    draw_prediction_panel(&left, dataset, before, "Antes do treino — weight=0, bias=0", false);
    draw_prediction_panel(
        &right,
        dataset,
        after,
        &format!(
            "Depois de 1000 epochs — weight={:.2}, bias={:.2}",
            after.weight, after.bias
        ),
        false,
    );

    root.present().unwrap();
    println!("Salvo em {path}");
}
