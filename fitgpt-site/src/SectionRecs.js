import MultiBar from './MultiBar';

function SectionRecs() {
	return (
		<div className="grid items-center gap-6 lg:grid-cols-[1fr_500px] lg:gap-12 xl:grid-cols-[1fr_600px]">
			<img
				alt="Placeholder"
				className="mx-auto aspect-video overflow-hidden rounded-xl object-cover sm:w-full"
				height="400"
				src={require("./assets/coffindance.png")}
				width="600"
			/>
			<div className="space-y-4">
				<h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
					Detailed Recommendations
				</h2>
				<p className="max-w-[600px] text-gray-400 md:text-xl">
					FitGPT provides targeted recommendations, predicting colors and probabilities
					for each article of clothing.
				</p>
				
				<div className="flex flex-col gap-4">
					<MultiBar article={"shirt"} data={{
						"white": 0.78,
						"black": 0.15
					}} />
					<MultiBar article={"outerwear"} data={{
						"black": 0.39,
						"white": 0.24,
						"cream": 0.15
					}} />
					<MultiBar article={"pants"} data={{
						"light_blue": 0.17,
						"dark_blue": 0.14,
						"white": 0.13
					}} />
					<MultiBar article={"shoes"} data={{
						"black": 0.57,
						"light_brown": 0.28
					}} />
				</div>
			</div>
		</div>
	)
}

export default SectionRecs;
