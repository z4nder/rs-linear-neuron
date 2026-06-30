mod plots;

pub struct Neuron {
    pub weight: f64,
    pub bias: f64,
}

impl Neuron {
    pub fn predict(&self, x: f64) -> f64 {
        self.weight * x + self.bias
    }

    pub fn train(&mut self, dataset: &[(f64, f64)], epochs: usize) {
        for epoch in 0..epochs {
            let mut total_error = 0.0;

            for (energy, actual) in dataset {
                let predicted = self.predict(*energy);
                let error = predicted - actual;
                total_error += error.abs();

                if error > 0.0 {
                    self.weight -= 0.01;
                    self.bias -= 0.01;
                } else if error < 0.0 {
                    self.weight += 0.01;
                    self.bias += 0.01;
                }
            }

            if epoch % 100 == 0 {
                println!(
                    "epoch {:>4}  weight: {:>6.3}  bias: {:>6.3}  erro total: {:>8.3}",
                    epoch, self.weight, self.bias, total_error
                );
            }
        }
    }
}

fn main() {
    let dataset: Vec<(f64, f64)> = vec![
        (10.0, 28.0),
        (20.0, 38.0),
        (30.0, 47.0),
        (40.0, 56.0),
        (50.0, 64.0),
        (60.0, 74.0),
        (70.0, 82.0),
        (80.0, 91.0),
        (90.0, 100.0),
    ];

    let mut neuron = Neuron {
        weight: 0.0,
        bias: 0.0,
    };
    plots::plot_slopes("assets/01_weight.png");
    plots::plot_biases("assets/01_bias.png");
    plots::plot_equation_comparison("assets/01_equation.png");
    plots::plot_error(&dataset, &neuron, "assets/01_error.png");
    plots::plot_error(&dataset, &neuron, "assets/01_error_before.png");
    plots::plot_error(
        &dataset,
        &Neuron {
            weight: 2.0,
            bias: 3.0,
        },
        "assets/01_error_example.png",
    );

    println!("\n--- training ---\n");
    neuron.train(&dataset, 1000);

    println!("\n--- resultado ---\n");
    println!(
        "{:<10} {:<10} {:<12} {:<10}",
        "energy", "actual", "predicted", "error"
    );
    println!("{}", "-".repeat(44));
    for (energy, actual) in &dataset {
        let predicted = neuron.predict(*energy);
        let error = predicted - actual;
        println!(
            "{:<10.1} {:<10.1} {:<12.1} {:<10.1}",
            energy, actual, predicted, error
        );
    }

    plots::plot_error(&dataset, &neuron, "assets/01_error_after.png");
    plots::plot_comparison(
        &dataset,
        &Neuron {
            weight: 0.0,
            bias: 0.0,
        },
        &neuron,
        "assets/01_comparison.png",
    );
}
