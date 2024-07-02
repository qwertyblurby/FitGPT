function DemoData(demoStatus) {
	if (demoStatus === "sample") {
		return {
			imageSrc: require("./sample.png"),
			results: {
				"shirt": {"white": 0.78, "black": 0.15},
				"outerwear": {"black": 0.39, "white": 0.24, "cream": 0.15},
				"pants": {"light_blue": 0.17, "dark_blue": 0.14, "white": 0.13},
				"shoes": {"black": 0.57, "light_brown": 0.28}
			}
		}
	}
	
	if (demoStatus === "demo_1") {
		const results = require("./demo_1_results.json");
		return {
			imageSrc: require("./demo_1.jpg"),
			results: results
		}
	}
	
	if (demoStatus === "demo_2") {
		const results = require("./demo_2_results.json");
		return {
			imageSrc: require("./demo_2.jpg"),
			results: results
		}
	}
}

export default DemoData;
