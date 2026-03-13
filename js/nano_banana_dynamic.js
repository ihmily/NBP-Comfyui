import { app } from "../../scripts/app.js";

console.log("🍌 [NanoBananaDynamicInputs] Executing JS Extension file...");

app.registerExtension({
    name: "NanoBanana.DynamicInputs",
    async nodeCreated(node) {
        console.log("🍌 [NanoBananaDynamicInputs] Node created:", node.comfyClass, node.type, node.name);

        // ComfyUI sometimes uses node.comfyClass and sometimes node.type
        if (node.comfyClass === "NanoBananaGenerate" || node.type === "NanoBananaGenerate") {
            console.log("🍌 [NanoBananaDynamicInputs] Matched NanoBananaGenerate node, hooking connections...");

            const onConnectionsChange = node.onConnectionsChange;
            node.onConnectionsChange = function (type, index, connected, link_info) {
                if (onConnectionsChange) {
                    onConnectionsChange.apply(this, arguments);
                }

                // If not an input, ignore
                if (type !== 1) return;

                console.log("🍌 [NanoBananaDynamicInputs] Connection changed. Type:", type, "Index:", index, "Connected:", connected);

                const PREFIX = "reference_image_";
                const MAX_INPUTS = 14;

                let imageInputs = [];
                for (let i = 0; i < this.inputs.length; i++) {
                    if (this.inputs[i].name.startsWith(PREFIX)) {
                        imageInputs.push(this.inputs[i]);
                    }
                }

                if (imageInputs.length === 0) return;

                let emptySlots = 0;
                for (let i = 0; i < imageInputs.length; i++) {
                    if (!imageInputs[i].link) {
                        emptySlots++;
                    }
                }

                console.log("🍌 [NanoBananaDynamicInputs] Current dynamic slots:", imageInputs.length, "Empty:", emptySlots);

                if (emptySlots === 0 && imageInputs.length < MAX_INPUTS) {
                    this.addInput(PREFIX + (imageInputs.length + 1), "IMAGE");
                    console.log("🍌 [NanoBananaDynamicInputs] Added new input slot.");
                }
                else if (emptySlots > 1) {
                    let removed = 0;
                    for (let i = this.inputs.length - 1; i >= 0; i--) {
                        if (this.inputs[i].name.startsWith(PREFIX) && !this.inputs[i].link) {
                            this.removeInput(i);
                            removed++;
                            console.log("🍌 [NanoBananaDynamicInputs] Removed extra empty slot.", i);
                            if (emptySlots - removed <= 1) {
                                break;
                            }
                        }
                    }
                }

                let count = 1;
                for (let i = 0; i < this.inputs.length; i++) {
                    if (this.inputs[i].name.startsWith(PREFIX)) {
                        this.inputs[i].name = PREFIX + count;
                        this.inputs[i].label = PREFIX + count;
                        count++;
                    }
                }

                if (this.computeSize) {
                    this.setSize(this.computeSize());
                }
                app.graph.setDirtyCanvas(true, true);
            };
        }
    }
});
