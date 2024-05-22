import Floater from './Floater';

function SectionCaps() {
	const floaters = [
		[
			"Blazing Fast",
			"FitGPT processes images in a fraction of a second, delivering lightning-fast results.",
			"BoltIcon"
		],
		[
			"Efficient Architecture",
			"FitGPT's architecture is simple and robust, being compatible with a wide range of hardware.",
			"MinimizeIcon"
		],
		[
			"Detailed Predictions",
			"FitGPT provides detailed probability distributions for each article of clothing, giving you insights into the model's confidence.",
			"ShirtIcon"
		],
		[
			"Automatic Person Detection",
			"FitGPT uses computer vision algorithms to automatically locate people within images, enabling more accurate classification.",
			"PersonStandingIcon"
		]
	]
	
	return (
		<div className="space-y-6">
			<h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">Model Capabilities</h2>
			<div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
				{floaters.map(([titleText, description, iconSrc]) => (
					<Floater
						key={titleText}
						titleText={titleText}
						description={description}
						iconSrc={iconSrc}
					/>
				))}
			</div>
		</div>
	)
}

export default SectionCaps;
