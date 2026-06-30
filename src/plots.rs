use crate::Neuron;
use plotters::prelude::*;
use std::fs;
use std::path::Path;

fn prepare_output_path(path: &str) {
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).unwrap();
        }
    }
}

pub fn plot_slopes(path: &str) {
    prepare_output_path(path);
    let root = BitMapBackend::new(path, (700, 500)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Efeito do weight (m) — mesmo bias, inclinações diferentes",
            ("sans-serif", 18),
        )
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..10f64, -5f64..25f64)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("x (entrada)")
        .y_desc("y (saída)")
        .draw()
        .unwrap();

    let lines: &[(f64, f64, &str, RGBColor)] = &[
        (0.5, 0.0, "m=0.5, b=0", RED),
        (1.0, 0.0, "m=1.0, b=0", BLUE),
        (2.0, 0.0, "m=2.0, b=0", GREEN),
    ];

    for (m, b, label, color) in lines {
        let points: Vec<(f64, f64)> = (0..=100)
            .map(|i| {
                let x = i as f64 * 0.1;
                (x, m * x + b)
            })
            .collect();

        chart
            .draw_series(LineSeries::new(points, color.stroke_width(2)))
            .unwrap()
            .label(*label)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
            });
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(BLACK)
        .background_style(WHITE.mix(0.9))
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Salvo em {path}");
}

pub fn plot_biases(path: &str) {
    prepare_output_path(path);
    let root = BitMapBackend::new(path, (700, 500)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Efeito do bias (b) — mesmo weight, deslocamentos diferentes",
            ("sans-serif", 18),
        )
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..10f64, -2f64..22f64)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("x (entrada)")
        .y_desc("y (saída)")
        .draw()
        .unwrap();

    let lines: &[(f64, f64, &str, RGBColor)] = &[
        (1.0, 0.0, "m=1, b=0", RED),
        (1.0, 5.0, "m=1, b=5", BLUE),
        (1.0, 10.0, "m=1, b=10", GREEN),
    ];

    for (m, b, label, color) in lines {
        let points: Vec<(f64, f64)> = (0..=100)
            .map(|i| {
                let x = i as f64 * 0.1;
                (x, m * x + b)
            })
            .collect();

        chart
            .draw_series(LineSeries::new(points, color.stroke_width(2)))
            .unwrap()
            .label(*label)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
            });
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(BLACK)
        .background_style(WHITE.mix(0.9))
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Salvo em {path}");
}

pub fn plot_error(dataset: &[(f64, f64)], neuron: &Neuron, path: &str) {
    prepare_output_path(path);
    let root = BitMapBackend::new(path, (700, 500)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let caption = format!("Neurônio — weight={}, bias={}", neuron.weight, neuron.bias);

    let mut chart = ChartBuilder::on(&root)
        .caption(&caption, ("sans-serif", 18))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..100f64, -10f64..110f64)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("energia")
        .y_desc("distância")
        .draw()
        .unwrap();

    let line: Vec<(f64, f64)> = (0..=100)
        .map(|x| (x as f64, neuron.predict(x as f64)))
        .collect();

    let legend_label = format!("previsão (w={}, b={})", neuron.weight, neuron.bias);
    chart
        .draw_series(LineSeries::new(line, BLUE.stroke_width(2)))
        .unwrap()
        .label(legend_label)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(2)));

    let error_color = RGBColor(255, 140, 0);

    for (energy, actual) in dataset {
        let predicted = neuron.predict(*energy);
        let error = predicted - actual;
        let mid_y = (predicted + actual) / 2.0;

        chart
            .draw_series(LineSeries::new(
                vec![(*energy, predicted), (*energy, *actual)],
                error_color.stroke_width(2),
            ))
            .unwrap();

        chart
            .draw_series(std::iter::once(Text::new(
                format!("{:.1}", error),
                (*energy + 1.5, mid_y),
                ("sans-serif", 10).into_font().color(&error_color),
            )))
            .unwrap();
    }

    chart
        .draw_series(std::iter::once(Circle::new((0.0, -999.0), 0, TRANSPARENT)))
        .unwrap()
        .label("erro a reduzir")
        .legend(move |(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], error_color.stroke_width(2))
        });

    chart
        .draw_series(
            dataset
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 5, RED.filled())),
        )
        .unwrap()
        .label("valores reais (gabarito)")
        .legend(|(x, y)| Circle::new((x + 10, y), 5, RED.filled()));

    chart
        .draw_series(dataset.iter().map(|(x, y)| {
            Text::new(
                format!("({}, {})", x, y),
                (*x + 2.0, *y + 3.0),
                ("sans-serif", 11).into_font().color(&BLACK),
            )
        }))
        .unwrap();

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(BLACK)
        .background_style(WHITE.mix(0.9))
        .draw()
        .unwrap();

    root.present().unwrap();
    println!("Salvo em {path}");
}

fn draw_panel(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    dataset: &[(f64, f64)],
    neuron: &Neuron,
    title: &str,
) {
    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 15))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..100f64, -10f64..110f64)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("energia")
        .y_desc("distância")
        .draw()
        .unwrap();

    let line: Vec<(f64, f64)> = (0..=100)
        .map(|x| (x as f64, neuron.predict(x as f64)))
        .collect();

    chart
        .draw_series(LineSeries::new(line, BLUE.stroke_width(2)))
        .unwrap()
        .label("previsão")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.stroke_width(2)));

    let error_color = RGBColor(255, 140, 0);
    for (energy, actual) in dataset {
        let predicted = neuron.predict(*energy);
        let error = predicted - actual;
        let mid_y = (predicted + actual) / 2.0;

        chart
            .draw_series(LineSeries::new(
                vec![(*energy, predicted), (*energy, *actual)],
                error_color.stroke_width(2),
            ))
            .unwrap();

        chart
            .draw_series(std::iter::once(Text::new(
                format!("{:.1}", error),
                (*energy + 1.5, mid_y),
                ("sans-serif", 10).into_font().color(&error_color),
            )))
            .unwrap();
    }

    chart
        .draw_series(std::iter::once(Circle::new((0.0, -999.0), 0, TRANSPARENT)))
        .unwrap()
        .label("erro")
        .legend(move |(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], error_color.stroke_width(2))
        });

    chart
        .draw_series(
            dataset
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 5, RED.filled())),
        )
        .unwrap()
        .label("gabarito")
        .legend(|(x, y)| Circle::new((x + 10, y), 5, RED.filled()));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .border_style(BLACK)
        .background_style(WHITE.mix(0.9))
        .draw()
        .unwrap();
}

pub fn plot_equation_comparison(path: &str) {
    prepare_output_path(path);
    let root = BitMapBackend::new(path, (1400, 500)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let (left, right) = root.split_horizontally(700);

    // painel esquerdo — efeito do weight
    {
        let mut chart = ChartBuilder::on(&left)
            .caption(
                "Efeito do weight (m) — mesma origem, inclinações diferentes",
                ("sans-serif", 15),
            )
            .margin(30)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0f64..10f64, -5f64..25f64)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("x")
            .y_desc("y")
            .draw()
            .unwrap();

        for (m, b, label, color) in [
            (0.5, 0.0, "m=0.5, b=0", RED),
            (1.0, 0.0, "m=1.0, b=0", BLUE),
            (2.0, 0.0, "m=2.0, b=0", GREEN),
        ] {
            let points: Vec<(f64, f64)> = (0..=100)
                .map(|i| {
                    let x = i as f64 * 0.1;
                    (x, m * x + b)
                })
                .collect();
            chart
                .draw_series(LineSeries::new(points, color.stroke_width(2)))
                .unwrap()
                .label(label)
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
                });
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .border_style(BLACK)
            .background_style(WHITE.mix(0.9))
            .draw()
            .unwrap();
    }

    {
        let mut chart = ChartBuilder::on(&right)
            .caption(
                "Efeito do bias (b) — mesma inclinação, deslocamentos diferentes",
                ("sans-serif", 15),
            )
            .margin(30)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(0f64..10f64, -2f64..22f64)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("x")
            .y_desc("y")
            .draw()
            .unwrap();

        for (m, b, label, color) in [
            (1.0, 0.0, "m=1, b=0", RED),
            (1.0, 5.0, "m=1, b=5", BLUE),
            (1.0, 10.0, "m=1, b=10", GREEN),
        ] {
            let points: Vec<(f64, f64)> = (0..=100)
                .map(|i| {
                    let x = i as f64 * 0.1;
                    (x, m * x + b)
                })
                .collect();
            chart
                .draw_series(LineSeries::new(points, color.stroke_width(2)))
                .unwrap()
                .label(label)
                .legend(move |(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], color.stroke_width(2))
                });
        }

        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .border_style(BLACK)
            .background_style(WHITE.mix(0.9))
            .draw()
            .unwrap();
    }

    root.present().unwrap();
    println!("Salvo em {path}");
}

pub fn plot_comparison(dataset: &[(f64, f64)], before: &Neuron, after: &Neuron, path: &str) {
    prepare_output_path(path);
    let root = BitMapBackend::new(path, (1400, 500)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let (left, right) = root.split_horizontally(700);

    draw_panel(&left, dataset, before, "Antes do treino — weight=0, bias=0");
    draw_panel(
        &right,
        dataset,
        after,
        &format!(
            "Depois de 1000 epochs — weight={:.2}, bias={:.2}",
            after.weight, after.bias
        ),
    );

    root.present().unwrap();
    println!("Salvo em {path}");
}
