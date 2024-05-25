import MultiBar from './MultiBar';
import { articleList } from './ModelData';

function SectionResults({ results }) {
	if (results === null) {
		return (
			<div className="flex flex-col items-center justify-center">
				<span className="loading loading-dots loading-lg"></span>
				<p className="mt-2 text-gray-500">Processing image</p>
			</div>
		)
	}
	
	return (
		<div className="space-y-6">
			<h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
				FitGPT Recommendations
			</h2>
			<div className="flex flex-col space-y-4">
				{articleList.map(article => 
					<MultiBar key={article} article={article} data={results[article]}/>
				)}
			</div>
		</div>
	)
}

export default SectionResults;
