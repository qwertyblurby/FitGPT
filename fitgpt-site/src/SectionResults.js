import FancyBar from './FancyBar';

function SectionResults() {
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
					Visualize Classification Results
				</h2>
				<p className="max-w-[600px] text-gray-400 md:text-xl">
					FitGPT provides targeted recommendations, predicting colors and probabilities
					for each article of clothing.
				</p>
				
				<div className="flex flex-col gap-4">
					<div>
						<p>Shirt</p>
						<FancyBar color="white" percent={78} />
					</div>
					<div>
						<p>Outerwear</p>
						<FancyBar color="black" percent={39} />
					</div>
					<div>
						<p>Shirt</p>
						<FancyBar color="light_blue" percent={17} />
					</div>
					<div>
						<p>Shoes</p>
						<FancyBar color="black" percent={57} />
					</div>
				</div>
			</div>
		</div>
	)
}

export default SectionResults;
