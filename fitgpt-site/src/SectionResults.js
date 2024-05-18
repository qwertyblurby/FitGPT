import FancyBar from './FancyBar';
import { colorList, articleList, articleMapping } from './ModelData';

function SectionResults({ results }) {
	if (results === null) {
		return (
			<div className="flex flex-col items-center justify-center">
				<span className="loading loading-dots loading-lg"></span>
				<p className="mt-2 text-gray-500">Processing image</p>
			</div>
		)
	}
	
	const sortedStack = (article) => {
		const sortedColors = colorList.map(color => (
			{
				color,
				percent: results[article][color] || 0
			}
		));
		sortedColors.sort((a, b) => (b.percent - a.percent));
		
		return (
			<div key={article} className="flex flex-col space-y-4">
				<h3 className="text-xl font-semibold">{articleMapping[article]}</h3>
				{sortedColors.map(({ color, percent }) => (
					<div key={color} className="w-full">
						<FancyBar color={color} percent={percent*100} />
					</div>
				))}
			</div>
		)
	};
	
	return (
		<div className="space-y-6">
			<h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
				FitGPT Recommendations
			</h2>
			<div className="grid grid-cols-1 gap-12 sm:grid-cols-2 lg:grid-cols-4">
				{articleList.map(sortedStack)}
			</div>
		</div>
	)
}

export default SectionResults;
