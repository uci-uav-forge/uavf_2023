extends Camera


# Declare member variables here. Examples:
# var a = 2
# var b = "text"


# Called when the node enters the scene tree for the first time.
func _ready():
	pass
var first = true

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	if first:
		get_viewport().set_clear_mode(Viewport.CLEAR_MODE_ONLY_NEXT_FRAME)
		yield(VisualServer, "frame_post_draw")
		first = false
		var image = get_viewport().get_texture().get_data()
		image.save_png("/home/eric/Code/forge-godot/roll-23-deg.png")
		
