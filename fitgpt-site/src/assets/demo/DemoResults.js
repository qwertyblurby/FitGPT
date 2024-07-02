import MultiBar from './MultiBar';
import { articleList } from './ModelData';

function SectionResults({ results, demoStatus }) {
	if (demoStatus === "loading") {
		return (
			<div className="bg-gray-800 rounded-lg flex-grow p-6 flex flex-col items-center justify-center space-y-4">
				<span className="loading loading-dots loading-lg"></span>
				<p className="text-gray-500">Processing image</p>
			</div>
		)
	}
	
	if (demoStatus === "error") {
		return (
			<div className="bg-gray-800 rounded-lg flex-grow p-6">
				<p className="text-red-400 text-center">
					An error occurred while processing your request. The demo server may be unavailable.
				</p>
			</div>
		)
	}
	
	return (
		<div className="bg-gray-800 rounded-lg p-6 flex-grow">
			<h3 className="text-2xl font-bold tracking-tighter mb-5 sm:text-3xl md:text-4xl">
				{demoStatus === "done" ? "FitGPT Recommendations" : "Example Recommendations"}
			</h3>
			<div className="flex flex-col space-y-4">
				{articleList.map(article => 
					<MultiBar key={article} article={article} data={results[article]}/>
				)}
			</div>
		</div>
	)
}

export default SectionResults;
