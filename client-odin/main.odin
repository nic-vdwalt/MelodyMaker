package main

import "core:encoding/json"
import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:strconv"
import "core:strings"
import rl "ingot:gfx"
import "ingot:prefs"
import "ingot:sys"
import "ingot:ui"

WINDOW_WIDTH :: 1180
WINDOW_HEIGHT :: 760
WINDOW_TITLE :: "Melody Trainer"
PREFS_APP :: "melody-trainer"
PREFS_FILE :: "settings.json"
STATUS_FILE :: ".melody-training-status.json"
MAX_LOSS_POINTS :: 256

Train_Status :: struct {
	phase:           string,
	epoch:           int,
	epochs:          int,
	progress:        f32,
	loss:            f64,
	val_loss:        f64,
	files_scanned:   int,
	files_total:     int,
	windows:         int,
	current_file:    string,
	message:         string,
	model_path:      string,
	seed_path:       string,
	elapsed_seconds: f64,
}

Settings :: struct {
	project_root: string,
	midi_dir:     string,
	model_path:   string,
	seed_path:    string,
	epochs:       string,
	batch_size:   string,
	seq_len:      string,
}

App :: struct {
	form:          ui.Ui,
	project_root:  ui.Input_Box,
	midi_dir:      ui.Input_Box,
	model_path:    ui.Input_Box,
	seed_path:     ui.Input_Box,
	epochs:        ui.Input_Box,
	batch_size:    ui.Input_Box,
	seq_len:       ui.Input_Box,
	process:       os.Process,
	has_process:   bool,
	status:        Train_Status,
	status_path:   string,
	error_message: string,
	losses:        [MAX_LOSS_POINTS]f32,
	val_losses:    [MAX_LOSS_POINTS]f32,
	loss_count:    int,
	last_epoch:    int,
}

pacer: ui.Frame_Pacer
ui_runtime: ui.Ui_Runtime
ui_frame: ui.Ui_Frame
app: App

set_input :: proc(box: ^ui.Input_Box, value: string) {
	assert(box != nil)
	ui.input_box_reset(box)
	strings.write_string(&box.sb, value)
	box.st.cursor = len(value)
	assert(ui.input_box_text(box) == value)
}

input_int :: proc(box: ^ui.Input_Box, fallback: int) -> int {
	assert(box != nil)
	value, ok := strconv.parse_int(strings.trim_space(ui.input_box_text(box)))
	if !ok || value < 1 do return fallback
	return value
}

app_init :: proc(state: ^App) {
	assert(state != nil)
	set_input(&state.project_root, "..")
	set_input(&state.midi_dir, "Rock_Music_Midi")
	set_input(&state.model_path, "melody_model.h5")
	set_input(&state.seed_path, "seed.pkl")
	set_input(&state.epochs, "20")
	set_input(&state.batch_size, "64")
	set_input(&state.seq_len, "60")
	settings_load(state)
	assert(input_int(&state.epochs, 20) > 0)
}

app_destroy :: proc(state: ^App) {
	assert(state != nil)
	training_stop(state)
	settings_save(state)
	ui.input_box_destroy(&state.project_root)
	ui.input_box_destroy(&state.midi_dir)
	ui.input_box_destroy(&state.model_path)
	ui.input_box_destroy(&state.seed_path)
	ui.input_box_destroy(&state.epochs)
	ui.input_box_destroy(&state.batch_size)
	ui.input_box_destroy(&state.seq_len)
	delete(state.status_path)
	delete(state.status.phase)
	delete(state.status.current_file)
	delete(state.status.message)
	delete(state.status.model_path)
	delete(state.status.seed_path)
	delete(state.error_message)
}

settings_load :: proc(state: ^App) {
	assert(state != nil)
	data, ok := prefs.read(PREFS_APP, PREFS_FILE)
	if !ok do return
	settings: Settings
	if json.unmarshal(data, &settings) != nil do return
	if settings.project_root != "" do set_input(&state.project_root, settings.project_root)
	if settings.midi_dir != "" do set_input(&state.midi_dir, settings.midi_dir)
	if settings.model_path != "" do set_input(&state.model_path, settings.model_path)
	if settings.seed_path != "" do set_input(&state.seed_path, settings.seed_path)
	if settings.epochs != "" do set_input(&state.epochs, settings.epochs)
	if settings.batch_size != "" do set_input(&state.batch_size, settings.batch_size)
	if settings.seq_len != "" do set_input(&state.seq_len, settings.seq_len)
	delete(settings.project_root)
	delete(settings.midi_dir)
	delete(settings.model_path)
	delete(settings.seed_path)
	delete(settings.epochs)
	delete(settings.batch_size)
	delete(settings.seq_len)
}

settings_save :: proc(state: ^App) {
	assert(state != nil)
	settings := Settings {
		project_root = ui.input_box_text(&state.project_root),
		midi_dir     = ui.input_box_text(&state.midi_dir),
		model_path   = ui.input_box_text(&state.model_path),
		seed_path    = ui.input_box_text(&state.seed_path),
		epochs       = ui.input_box_text(&state.epochs),
		batch_size   = ui.input_box_text(&state.batch_size),
		seq_len      = ui.input_box_text(&state.seq_len),
	}
	data, err := json.marshal(settings, {pretty = true}, context.temp_allocator)
	if err == nil do _ = prefs.write(PREFS_APP, PREFS_FILE, data)
}

set_error :: proc(state: ^App, message: string) {
	assert(state != nil)
	delete(state.error_message)
	state.error_message = strings.clone(message)
}

training_start :: proc(state: ^App) {
	assert(state != nil)
	if state.has_process do return
	root := strings.trim_space(ui.input_box_text(&state.project_root))
	if root == "" {
		set_error(state, "Choose the MelodyMaker project directory.")
		return
	}
	absolute_root, abs_err := filepath.abs(root)
	if abs_err != nil {
		set_error(state, "The project directory could not be resolved.")
		return
	}
	python_path, _ := filepath.join(
		{absolute_root, ".venv", "bin", "python"},
		context.temp_allocator,
	)
	when ODIN_OS == .Windows {
		python_path, _ = filepath.join(
			{absolute_root, ".venv", "Scripts", "python.exe"},
			context.temp_allocator,
		)
	}
	train_path, _ := filepath.join({absolute_root, "src", "train.py"}, context.temp_allocator)
	status_path, _ := filepath.join({absolute_root, STATUS_FILE}, context.temp_allocator)
	if !os.exists(python_path) || !os.exists(train_path) {
		set_error(
			state,
			"Python environment or src/train.py was not found in the project directory.",
		)
		return
	}
	os.remove(status_path)
	command := []string {
		python_path,
		train_path,
		"--midi_dir",
		ui.input_box_text(&state.midi_dir),
		"--melody_path",
		ui.input_box_text(&state.model_path),
		"--seed_path",
		ui.input_box_text(&state.seed_path),
		"--epochs",
		ui.input_box_text(&state.epochs),
		"--batch_size",
		ui.input_box_text(&state.batch_size),
		"--seq_len",
		ui.input_box_text(&state.seq_len),
		"--status_path",
		status_path,
	}
	process, process_err := os.process_start({command = command, working_dir = absolute_root})
	if process_err != nil {
		set_error(state, fmt.tprintf("Could not start training: %v", process_err))
		return
	}
	delete(state.status_path)
	state.status_path = strings.clone(status_path)
	state.process = process
	state.has_process = true
	state.loss_count = 0
	state.last_epoch = 0
	state.status = {}
	set_error(state, "")
	settings_save(state)
	assert(state.has_process)
}

training_stop :: proc(state: ^App) {
	assert(state != nil)
	if !state.has_process do return
	if err := os.process_terminate(state.process); err != nil {
		_ = os.process_kill(state.process)
	}
	_, _ = os.process_wait(state.process)
	state.has_process = false
	assert(!state.has_process)
}

status_replace :: proc(state: ^App, next: Train_Status) {
	assert(state != nil)
	delete(state.status.phase)
	delete(state.status.current_file)
	delete(state.status.message)
	delete(state.status.model_path)
	delete(state.status.seed_path)
	state.status = next
	if next.epoch > state.last_epoch && next.epoch > 0 && state.loss_count < MAX_LOSS_POINTS {
		state.losses[state.loss_count] = f32(next.loss)
		state.val_losses[state.loss_count] = f32(next.val_loss)
		state.loss_count += 1
		state.last_epoch = next.epoch
	}
}

training_poll :: proc(state: ^App) {
	assert(state != nil)
	if state.status_path != "" {
		if data, read_err := os.read_entire_file(state.status_path, context.temp_allocator);
		   read_err == nil {
			next: Train_Status
			if json.unmarshal(data, &next) == nil do status_replace(state, next)
		}
	}
	if !state.has_process do return
	process_state, wait_err := os.process_wait(state.process, 0)
	if wait_err == .Timeout do return
	state.has_process = false
	if process_state.exit_code != 0 && state.status.phase != "error" {
		set_error(
			state,
			fmt.tprintf(
				"Training exited with code %d. Check the terminal output.",
				process_state.exit_code,
			),
		)
	}
}

input_field :: proc(
	state: ^App,
	box: ^ui.Input_Box,
	label, placeholder: string,
	id: ui.Focus_Id,
	x, y, w: i32,
) {
	assert(state != nil && box != nil)
	ui.draw_text_frame(
		&ui_frame,
		strings.clone_to_cstring(label, context.temp_allocator),
		x,
		y,
		ui.FONT_SIZE_LABEL,
		ui.ui_frame_theme(&ui_frame).fg_label,
	)
	focus := ui.ui_focus(&state.form, id)
	ui.focus_opt_click(
		&ui_frame,
		focus,
		x,
		y + ui.ui_frame_sc(&ui_frame, 21),
		w,
		ui.ui_frame_sc(&ui_frame, 36),
	)
	_ = ui.input_at(
		&ui_frame,
		x,
		y + ui.ui_frame_sc(&ui_frame, 21),
		w,
		ui.ui_frame_sc(&ui_frame, 36),
		box,
		placeholder,
		ui.focus_opt_focused(focus),
		semantics = {name = label, focus = focus.focus, focus_id = focus.id},
	)
}

metric_card :: proc(x, y, w, h: i32, label, value: string, color: rl.Color) {
	assert(w > 0 && h > 0)
	ui.draw_card_bg_frame(
		&ui_frame,
		{f32(x), f32(y), f32(w), f32(h)},
		ui.ui_frame_theme(&ui_frame).bg_panel,
		color,
		ui.ui_frame_sc(&ui_frame, 3),
	)
	ui.draw_text_frame(
		&ui_frame,
		strings.clone_to_cstring(label, context.temp_allocator),
		x + ui.ui_frame_sc(&ui_frame, 14),
		y + ui.ui_frame_sc(&ui_frame, 12),
		ui.FONT_SIZE_LABEL,
		ui.ui_frame_theme(&ui_frame).fg_label,
	)
	ui.draw_text_frame(
		&ui_frame,
		strings.clone_to_cstring(value, context.temp_allocator),
		x + ui.ui_frame_sc(&ui_frame, 14),
		y + ui.ui_frame_sc(&ui_frame, 37),
		ui.FONT_SIZE_LARGE,
		ui.ui_frame_theme(&ui_frame).fg_primary,
	)
}

loss_chart :: proc(state: ^App, x, y, w, h: i32) {
	assert(state != nil)
	ui.draw_card_bg_frame(
		&ui_frame,
		{f32(x), f32(y), f32(w), f32(h)},
		ui.ui_frame_theme(&ui_frame).bg_panel,
	)
	ui.draw_text_frame(
		&ui_frame,
		"LOSS HISTORY",
		x + ui.ui_frame_sc(&ui_frame, 16),
		y + ui.ui_frame_sc(&ui_frame, 14),
		ui.FONT_SIZE_LABEL,
		ui.ui_frame_theme(&ui_frame).fg_label,
	)
	plot_x := x + ui.ui_frame_sc(&ui_frame, 16)
	plot_y := y + ui.ui_frame_sc(&ui_frame, 42)
	plot_w := w - ui.ui_frame_sc(&ui_frame, 32)
	plot_h := h - ui.ui_frame_sc(&ui_frame, 60)
	rl.DrawRectangle(plot_x, plot_y, plot_w, plot_h, ui.ui_frame_theme(&ui_frame).bg_input)
	if state.loss_count < 2 {
		ui.draw_text_frame(
			&ui_frame,
			"Loss appears after the first epoch",
			plot_x + ui.ui_frame_sc(&ui_frame, 12),
			plot_y + ui.ui_frame_sc(&ui_frame, 12),
			ui.FONT_SIZE_NOTE,
			ui.ui_frame_theme(&ui_frame).fg_secondary,
		)
		return
	}
	maximum: f32 = 0.001
	for index in 0 ..< state.loss_count {
		maximum = max(maximum, state.losses[index], state.val_losses[index])
	}
	for index in 1 ..< state.loss_count {
		x1 := f32(plot_x) + f32(index - 1) / f32(state.loss_count - 1) * f32(plot_w)
		x2 := f32(plot_x) + f32(index) / f32(state.loss_count - 1) * f32(plot_w)
		y1 := f32(plot_y + plot_h) - state.losses[index - 1] / maximum * f32(plot_h)
		y2 := f32(plot_y + plot_h) - state.losses[index] / maximum * f32(plot_h)
		rl.DrawLineEx({x1, y1}, {x2, y2}, 2, ui.ui_frame_theme(&ui_frame).fg_accent)
		vy1 := f32(plot_y + plot_h) - state.val_losses[index - 1] / maximum * f32(plot_h)
		vy2 := f32(plot_y + plot_h) - state.val_losses[index] / maximum * f32(plot_h)
		rl.DrawLineEx({x1, vy1}, {x2, vy2}, 2, ui.ui_frame_theme(&ui_frame).fg_success)
	}
}

phase_label :: proc(state: ^App) -> string {
	assert(state != nil)
	if state.status.phase == "scanning" do return "Scanning MIDI dataset"
	if state.status.phase == "training" do return "Training model"
	if state.status.phase == "complete" do return "Training complete"
	if state.status.phase == "error" do return "Training failed"
	return "Ready to train"
}

frame :: proc() {
	free_all(context.temp_allocator)
	training_poll(&app)
	ui.ui_runtime_dpi_refresh(&ui_runtime)
	ui.ui_frame_begin(&ui_frame, &ui_runtime)
	rl.BeginDrawing()
	rl.ClearBackground(ui.ui_frame_theme(&ui_frame).bg_app)

	sw := rl.GetScreenWidth()
	sh := rl.GetScreenHeight()
	header_h := ui.ui_frame_sc(&ui_frame, 76)
	rl.DrawRectangle(0, 0, sw, header_h, ui.ui_frame_theme(&ui_frame).bg_panel)
	rl.DrawRectangle(0, header_h - 1, sw, 1, ui.ui_frame_theme(&ui_frame).border_subtle)
	ui.draw_text_frame(
		&ui_frame,
		"MELODY",
		ui.ui_frame_sc(&ui_frame, 28),
		ui.ui_frame_sc(&ui_frame, 18),
		ui.FONT_SIZE_LARGE,
		ui.ui_frame_theme(&ui_frame).fg_accent,
	)
	ui.draw_text_frame(
		&ui_frame,
		"Training Studio",
		ui.ui_frame_sc(&ui_frame, 28),
		ui.ui_frame_sc(&ui_frame, 43),
		ui.FONT_SIZE_NOTE,
		ui.ui_frame_theme(&ui_frame).fg_secondary,
	)
	status_color :=
		ui.ui_frame_theme(&ui_frame).fg_success if app.status.phase == "complete" else ui.ui_frame_theme(&ui_frame).fg_accent
	if app.status.phase == "error" do status_color = ui.ui_frame_theme(&ui_frame).fg_error
	status_text := phase_label(&app)
	pill_w := ui.ui_frame_sc(&ui_frame, 190)
	_ = ui.status_pill(
		status_text,
		sw - pill_w - ui.ui_frame_sc(&ui_frame, 28),
		ui.ui_frame_sc(&ui_frame, 26),
		ui.FONT_SIZE_NOTE,
		status_color,
	)

	pad := ui.ui_frame_sc(&ui_frame, 24)
	gap := ui.ui_frame_sc(&ui_frame, 18)
	left_w := min(ui.ui_frame_sc(&ui_frame, 390), sw / 3)
	content_y := header_h + pad
	content_h := sh - content_y - pad
	ui.draw_card_bg_frame(
		&ui_frame,
		{f32(pad), f32(content_y), f32(left_w), f32(content_h)},
		ui.ui_frame_theme(&ui_frame).bg_panel,
	)
	form_x := pad + ui.ui_frame_sc(&ui_frame, 18)
	form_w := left_w - ui.ui_frame_sc(&ui_frame, 36)
	form_y := content_y + ui.ui_frame_sc(&ui_frame, 18)
	ui.draw_text_frame(
		&ui_frame,
		"TRAINING CONFIGURATION",
		form_x,
		form_y,
		ui.FONT_SIZE_LABEL,
		ui.ui_frame_theme(&ui_frame).fg_label,
	)
	form_y += ui.ui_frame_sc(&ui_frame, 28)
	ui.ui_begin_frame(
		&app.form,
		&ui_frame,
		form_x,
		form_y,
		form_w,
		content_h - ui.ui_frame_sc(&ui_frame, 110),
		gap = 0,
	)
	input_field(
		&app,
		&app.project_root,
		"Project directory",
		"..",
		ui.focus_id(1),
		form_x,
		form_y,
		form_w - ui.ui_frame_sc(&ui_frame, 90),
	)
	if ui.btn_at(
		&ui_frame,
		form_x + form_w - ui.ui_frame_sc(&ui_frame, 80),
		form_y + ui.ui_frame_sc(&ui_frame, 21),
		ui.ui_frame_sc(&ui_frame, 80),
		ui.ui_frame_sc(&ui_frame, 36),
		"Choose",
	) {
		if path, ok := sys.open_file_dialog("Choose any file in MelodyMaker"); ok {
			set_input(&app.project_root, filepath.dir(path))
			delete(path)
		}
	}
	form_y += ui.ui_frame_sc(&ui_frame, 72)
	input_field(
		&app,
		&app.midi_dir,
		"MIDI dataset",
		"Rock_Music_Midi",
		ui.focus_id(2),
		form_x,
		form_y,
		form_w,
	)
	form_y += ui.ui_frame_sc(&ui_frame, 72)
	input_field(
		&app,
		&app.model_path,
		"Model output",
		"melody_model.h5",
		ui.focus_id(3),
		form_x,
		form_y,
		form_w,
	)
	form_y += ui.ui_frame_sc(&ui_frame, 72)
	input_field(
		&app,
		&app.seed_path,
		"Seed output",
		"seed.pkl",
		ui.focus_id(4),
		form_x,
		form_y,
		form_w,
	)
	form_y += ui.ui_frame_sc(&ui_frame, 72)
	third := (form_w - gap * 2) / 3
	input_field(&app, &app.epochs, "Epochs", "20", ui.focus_id(5), form_x, form_y, third)
	input_field(
		&app,
		&app.batch_size,
		"Batch",
		"64",
		ui.focus_id(6),
		form_x + third + gap,
		form_y,
		third,
	)
	input_field(
		&app,
		&app.seq_len,
		"Sequence",
		"60",
		ui.focus_id(7),
		form_x + (third + gap) * 2,
		form_y,
		third,
	)
	ui.ui_end(&app.form)
	button_y := content_y + content_h - ui.ui_frame_sc(&ui_frame, 62)
	if app.has_process {
		if ui.btn_at(&ui_frame, form_x, button_y, form_w, ui.ui_frame_sc(&ui_frame, 42), "Stop training", .Danger) do training_stop(&app)
	} else {
		if ui.btn_at(&ui_frame, form_x, button_y, form_w, ui.ui_frame_sc(&ui_frame, 42), "Start training", .Primary) do training_start(&app)
	}

	right_x := pad + left_w + gap
	right_w := sw - right_x - pad
	card_gap := ui.ui_frame_sc(&ui_frame, 14)
	card_w := (right_w - card_gap * 2) / 3
	metric_card(
		right_x,
		content_y,
		card_w,
		ui.ui_frame_sc(&ui_frame, 82),
		"EPOCH",
		fmt.tprintf(
			"%d / %d",
			app.status.epoch,
			max(app.status.epochs, input_int(&app.epochs, 20)),
		),
		ui.ui_frame_theme(&ui_frame).fg_accent,
	)
	metric_card(
		right_x + card_w + card_gap,
		content_y,
		card_w,
		ui.ui_frame_sc(&ui_frame, 82),
		"TRAIN LOSS",
		fmt.tprintf("%.4f", app.status.loss),
		ui.ui_frame_theme(&ui_frame).fg_tool,
	)
	metric_card(
		right_x + (card_w + card_gap) * 2,
		content_y,
		card_w,
		ui.ui_frame_sc(&ui_frame, 82),
		"VALIDATION",
		fmt.tprintf("%.4f", app.status.val_loss),
		ui.ui_frame_theme(&ui_frame).fg_success,
	)

	progress_y := content_y + ui.ui_frame_sc(&ui_frame, 100)
	ui.draw_card_bg_frame(
		&ui_frame,
		{f32(right_x), f32(progress_y), f32(right_w), f32(ui.ui_frame_sc(&ui_frame, 104))},
		ui.ui_frame_theme(&ui_frame).bg_panel,
	)
	ui.draw_text_frame(
		&ui_frame,
		strings.clone_to_cstring(status_text, context.temp_allocator),
		right_x + ui.ui_frame_sc(&ui_frame, 16),
		progress_y + ui.ui_frame_sc(&ui_frame, 14),
		ui.FONT_SIZE_BODY,
		ui.ui_frame_theme(&ui_frame).fg_primary,
	)
	progress_detail := "Configure the dataset and start a run."
	if app.status.phase == "scanning" {
		progress_detail = fmt.tprintf(
			"%d / %d files  •  %d windows  •  %s",
			app.status.files_scanned,
			app.status.files_total,
			app.status.windows,
			app.status.current_file,
		)
	} else if app.status.phase == "training" {
		progress_detail = fmt.tprintf(
			"Epoch %d of %d  •  %.0f seconds elapsed",
			app.status.epoch,
			app.status.epochs,
			app.status.elapsed_seconds,
		)
	} else if app.status.phase == "complete" {
		progress_detail = fmt.tprintf("Saved %s", app.status.model_path)
	} else if app.status.phase == "error" {
		progress_detail = app.status.message
	}
	ui.draw_text_frame(
		&ui_frame,
		strings.clone_to_cstring(progress_detail, context.temp_allocator),
		right_x + ui.ui_frame_sc(&ui_frame, 16),
		progress_y + ui.ui_frame_sc(&ui_frame, 41),
		ui.FONT_SIZE_NOTE,
		ui.ui_frame_theme(&ui_frame).fg_secondary,
	)
	ui.progress_bar(
		right_x + ui.ui_frame_sc(&ui_frame, 16),
		progress_y + ui.ui_frame_sc(&ui_frame, 73),
		right_w - ui.ui_frame_sc(&ui_frame, 32),
		ui.ui_frame_sc(&ui_frame, 10),
		app.status.progress,
		status_color,
	)
	if app.has_process do ui.spinner(right_x + right_w - ui.ui_frame_sc(&ui_frame, 30), progress_y + ui.ui_frame_sc(&ui_frame, 24), ui.ui_frame_scf(&ui_frame, 9))

	chart_y := progress_y + ui.ui_frame_sc(&ui_frame, 122)
	loss_chart(
		&app,
		right_x,
		chart_y,
		right_w,
		max(
			ui.ui_frame_sc(&ui_frame, 180),
			content_y + content_h - chart_y - ui.ui_frame_sc(&ui_frame, 72),
		),
	)
	footer_y := content_y + content_h - ui.ui_frame_sc(&ui_frame, 54)
	if app.error_message != "" {
		ui.draw_text_frame(
			&ui_frame,
			strings.clone_to_cstring(app.error_message, context.temp_allocator),
			right_x,
			footer_y,
			ui.FONT_SIZE_NOTE,
			ui.ui_frame_theme(&ui_frame).fg_error,
		)
	} else {
		footer := "TensorFlow is required in .venv before training can start."
		if app.status.phase == "complete" do footer = "Model and seed are ready for MIDI generation."
		ui.draw_text_frame(
			&ui_frame,
			strings.clone_to_cstring(footer, context.temp_allocator),
			right_x,
			footer_y,
			ui.FONT_SIZE_NOTE,
			ui.ui_frame_theme(&ui_frame).fg_secondary,
		)
	}

	ui.a11y_frame_end(&ui_frame)
	ui.ui_frame_end(&ui_frame)
	rl.EndDrawing()
	ui.pacer_frame(&pacer, app.has_process)
}

main :: proc() {
	when ODIN_OS == .Darwin {
		rl.SetConfigFlags({.WINDOW_RESIZABLE, .VSYNC_HINT, .WINDOW_HIGHDPI, .WINDOW_TRANSPARENT})
	} else {
		rl.SetConfigFlags({.WINDOW_RESIZABLE, .VSYNC_HINT})
	}
	rl.InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE)
	rl.SetWindowMinSize(960, 640)
	rl.SetExitKey(.KEY_NULL)
	ui.apply_window_style()
	ui.titlebar_init()
	ui.ui_runtime_init(&ui_runtime)
	ui.ui_runtime_apply_platform_dpi(&ui_runtime)
	_ = ui.a11y_init(&ui_runtime)
	pacer = ui.pacer_init(60, 15, 2.5)
	app_init(&app)
	rl.run(frame)
	app_destroy(&app)
	ui.ui_runtime_destroy(&ui_runtime)
	rl.CloseWindow()
}
