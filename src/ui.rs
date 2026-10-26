//! egui/eframe GUI implementation.

/// Main application state for egui/eframe.
#[derive(Default)]
pub struct App;

impl eframe::App for App {
    fn ui(&mut self, _ui: &mut eframe::egui::Ui, _frame: &mut eframe::Frame) {}
}
