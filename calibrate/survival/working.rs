use super::layout::SurvivalLayout;
use crate::calibrate::model::LinkFunction;
use ndarray::{Array1, Array2, Axis, s};

#[derive(Debug, Clone)]
pub struct WorkingState {
    pub eta: Array1<f64>,
    pub gradient: Array1<f64>,
    pub hessian: Array2<f64>,
    pub deviance: f64,
}

pub trait WorkingModel {
    fn update(&mut self, beta: &Array1<f64>) -> WorkingState;
}

pub struct WorkingModelGam {
    design: Array2<f64>,
    response: Array1<f64>,
    weights: Array1<f64>,
    link: LinkFunction,
}

impl WorkingModelGam {
    pub fn new(
        design: Array2<f64>,
        response: Array1<f64>,
        weights: Array1<f64>,
        link: LinkFunction,
    ) -> Self {
        Self {
            design,
            response,
            weights,
            link,
        }
    }
}

impl WorkingModel for WorkingModelGam {
    fn update(&mut self, beta: &Array1<f64>) -> WorkingState {
        let eta = self.design.dot(beta);
        let (mu, var, deviance) = match self.link {
            LinkFunction::Logit => {
                let mu = eta.map(|v| 1.0 / (1.0 + (-v).exp()));
                let var = mu.map(|m| m * (1.0 - m));
                let mut dev = 0.0;
                for ((&y, &m), &w) in self.response.iter().zip(mu.iter()).zip(self.weights.iter()) {
                    if y > 0.0 {
                        dev -= 2.0 * w * (m.ln());
                    }
                    if y < 1.0 {
                        dev -= 2.0 * w * ((1.0 - m).ln());
                    }
                }
                (mu, var, dev)
            }
            LinkFunction::Identity => {
                let mu = eta.clone();
                let var = Array1::ones(mu.len());
                let mut dev = 0.0;
                for ((&y, &m), &w) in self.response.iter().zip(mu.iter()).zip(self.weights.iter()) {
                    dev += w * (y - m).powi(2);
                }
                (mu, var, dev)
            }
        };

        let mut gradient = Array1::zeros(beta.len());
        let mut hessian = Array2::zeros((beta.len(), beta.len()));
        for i in 0..self.design.nrows() {
            let xi = self.design.row(i).to_owned();
            let residual = self.response[i] - mu[i];
            gradient += &(xi.clone() * (self.weights[i] * residual));
            let weight = self.weights[i] * var[i];
            hessian += &(xi
                .view()
                .to_owned()
                .insert_axis(Axis(1))
                .dot(&xi.insert_axis(Axis(0)))
                * weight);
        }

        WorkingState {
            eta,
            gradient,
            hessian,
            deviance,
        }
    }
}

pub struct WorkingModelSurvival {
    x_exit: Array2<f64>,
    x_entry: Array2<f64>,
    x_deriv_exit: Array2<f64>,
    event_target: Array1<f64>,
    sample_weight: Array1<f64>,
    barrier_weight: f64,
    barrier_scale: f64,
    derivative_epsilon: f64,
}

impl WorkingModelSurvival {
    pub fn new(
        layout: SurvivalLayout,
        event_target: Array1<u8>,
        sample_weight: Array1<f64>,
    ) -> Self {
        let baseline_cols = layout.baseline_exit.ncols();
        let static_cols = layout.static_covariates.ncols();
        let total_cols = baseline_cols + static_cols;
        let x_exit = ndarray::concatenate(
            Axis(1),
            &[layout.baseline_exit.view(), layout.static_covariates.view()],
        )
        .expect("concatenate should succeed")
        .to_owned();
        let x_entry = ndarray::concatenate(
            Axis(1),
            &[
                layout.baseline_entry.view(),
                layout.static_covariates.view(),
            ],
        )
        .expect("concatenate should succeed")
        .to_owned();
        let mut x_deriv_exit = Array2::zeros((layout.baseline_derivative_exit.nrows(), total_cols));
        x_deriv_exit
            .slice_mut(s![.., 0..baseline_cols])
            .assign(&layout.baseline_derivative_exit);
        Self {
            x_exit,
            x_entry,
            x_deriv_exit,
            event_target: event_target.map(|v| v as f64),
            sample_weight,
            barrier_weight: 10.0,
            barrier_scale: 1.0,
            derivative_epsilon: 1e-8,
        }
    }

    fn softplus(&self, x: f64) -> f64 {
        if x > 30.0 { x } else { (1.0 + x.exp()).ln() }
    }

    fn sigmoid(&self, x: f64) -> f64 {
        if x >= 0.0 {
            let z = (-x).exp();
            1.0 / (1.0 + z)
        } else {
            let z = x.exp();
            z / (1.0 + z)
        }
    }
}

impl WorkingModel for WorkingModelSurvival {
    fn update(&mut self, beta: &Array1<f64>) -> WorkingState {
        let eta_exit = self.x_exit.dot(beta);
        let eta_entry = self.x_entry.dot(beta);
        let deta_exit = self.x_deriv_exit.dot(beta);

        let h_exit = eta_exit.map(|v| v.exp());
        let h_entry = eta_entry.map(|v| v.exp());
        let delta_h = &h_exit - &h_entry;

        let mut gradient = Array1::zeros(beta.len());
        let mut hessian = Array2::zeros((beta.len(), beta.len()));
        let mut deviance = 0.0;

        for idx in 0..self.x_exit.nrows() {
            let w = self.sample_weight[idx];
            let target = self.event_target[idx];
            let eta_exit_i = eta_exit[idx];
            let h_exit_i = h_exit[idx];
            let h_entry_i = h_entry[idx];
            let delta_h_i = (delta_h[idx]).max(1e-12);
            let x_exit_row = self.x_exit.row(idx).to_owned();
            let x_entry_row = self.x_entry.row(idx).to_owned();
            let x_deriv_row = self.x_deriv_exit.row(idx).to_owned();
            let deta_i = deta_exit[idx];
            let guard = if deta_i > self.derivative_epsilon {
                1.0 / deta_i
            } else {
                0.0
            };
            let log_guard = if deta_i > self.derivative_epsilon {
                deta_i
            } else {
                self.derivative_epsilon
            };
            let ll = target * (eta_exit_i + log_guard.ln()) - delta_h_i;
            deviance -= 2.0 * w * ll;

            gradient += &(x_entry_row.clone() * (w * h_entry_i));
            gradient -= &(x_exit_row.clone() * (w * h_exit_i));
            gradient += &(x_exit_row.clone() * (w * target));
            gradient += &(x_deriv_row.clone() * (w * target * guard));

            let entry_outer = x_entry_row
                .view()
                .to_owned()
                .insert_axis(Axis(1))
                .dot(&x_entry_row.insert_axis(Axis(0)));
            let exit_outer = x_exit_row
                .view()
                .to_owned()
                .insert_axis(Axis(1))
                .dot(&x_exit_row.insert_axis(Axis(0)));
            let deriv_outer = x_deriv_row
                .view()
                .to_owned()
                .insert_axis(Axis(1))
                .dot(&x_deriv_row.insert_axis(Axis(0)));

            hessian += &(entry_outer * (w * h_entry_i));
            hessian -= &(exit_outer * (w * h_exit_i));
            hessian -= &(deriv_outer * (w * target * guard * guard));

            let z = -deta_i / self.barrier_scale;
            let sig = self.sigmoid(z);
            let barrier_grad = -self.barrier_weight * w * sig / self.barrier_scale;
            let barrier_hess = self.barrier_weight * w * sig * (1.0 - sig)
                / (self.barrier_scale * self.barrier_scale);
            gradient += &(x_deriv_row.clone() * barrier_grad);
            hessian += &(deriv_outer * barrier_hess);
            deviance += 2.0 * w * self.barrier_weight * self.softplus(z);
        }

        WorkingState {
            eta: eta_exit,
            gradient,
            hessian,
            deviance,
        }
    }
}
