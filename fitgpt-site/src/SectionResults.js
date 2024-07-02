import MultiBar from './MultiBar';
import { articleList } from './ModelData';

function SectionResults({ results, uploadedImage, demoStatus }) {
	if (demoStatus === "loading") {
		return (
			<div className="min-h-80 flex flex-col items-center justify-center space-y-4">
				<span className="loading loading-dots loading-lg"></span>
				<p className="text-gray-500">Processing image</p>
			</div>
		)
	}
	
	let imagePath;
	let header = "Example Recommendations";
	let displayResults;
	if (demoStatus === "sample") {
		displayResults = {
			"shirt": {"white": 0.78, "black": 0.15},
			"outerwear": {"black": 0.39, "white": 0.24, "cream": 0.15},
			"pants": {"light_blue": 0.17, "dark_blue": 0.14, "white": 0.13},
			"shoes": {"black": 0.57, "light_brown": 0.28}
		};
	} else if (demoStatus === "demo_1") {
		const data_demo_1 = require("./assets/demo/demo_1_results.json");
		displayResults = data_demo_1;
	} else if (demoStatus === "demo_2") {
		const data_demo_2 = require("./assets/demo/demo_2_results.json");
		displayResults = data_demo_2;
	} else {
		displayResults = results;
		header = "FitGPT Recommendations";
	}
	
	return (
		<>
		<h3 className="text-2xl font-bold tracking-tighter mb-5 sm:text-3xl md:text-4xl">
			{header}
		</h3>
		{/* <div className="grid items-center gap-6 lg:grid-cols-[300px_1fr] lg:gap-12 xl:grid-cols-[350px_1fr]"> */}
		{/* <img src={uploadedImage} alt="Uploaded" className="overflow-hidden rounded-xl object-cover h-auto sm:mb-5 lg:mb-auto sm:w-1/2 lg:w-full" /> */}
		<div className="flex flex-col space-y-4">
			{articleList.map(article => 
				<MultiBar key={article} article={article} data={displayResults[article]}/>
			)}
		</div>
		</>
	)
}

export default SectionResults;
