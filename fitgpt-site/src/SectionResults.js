import MultiBar from './MultiBar';
import { articleList } from './ModelData';

function SectionResults({ results, uploadedImage }) {
	if (results === null) {
		return (
			<div className="flex flex-col items-center justify-center">
				<span className="loading loading-dots loading-lg"></span>
				<p className="mt-2 text-gray-500">Processing image</p>
			</div>
		)
	}
	
	return (
		<>
		<h2 className="text-3xl font-bold tracking-tighter mb-5 sm:text-4xl md:text-5xl">
			FitGPT Recommendations
		</h2>
		<div className="grid items-center gap-6 lg:grid-cols-[300px_1fr] lg:gap-12 xl:grid-cols-[350px_1fr]">
			<img src={uploadedImage} alt="Uploaded" className="overflow-hidden rounded-xl object-cover h-auto sm:mb-5 lg:mb-auto sm:w-1/2 lg:w-full" />
			<div className="flex flex-col space-y-4">
				{articleList.map(article => 
					<MultiBar key={article} article={article} data={results[article]}/>
				)}
			</div>
		</div>
		</>
	)
}

export default SectionResults;
