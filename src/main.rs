#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::Result;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--cli") {
        return pasfoto::pipeline::run_cli();
    }

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([
                pasfoto::config::UI_INIT_WIDTH,
                pasfoto::config::UI_INIT_HEIGHT,
            ])
            .with_min_inner_size([
                pasfoto::config::UI_MIN_WIDTH,
                pasfoto::config::UI_MIN_HEIGHT,
            ]),
        ..Default::default()
    };

    eframe::run_native(
        pasfoto::config::APP_TITLE,
        options,
        Box::new(|cc| Ok(Box::new(pasfoto::ui::App::new(cc)))),
    )
    .map_err(|e| anyhow::anyhow!("Failed to run UI: {e}"))
}
