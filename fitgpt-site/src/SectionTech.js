import Floater from './Floater';

function SectionTech() {
	const floaters = [
		[
			"Prediction Process",
			"Input images go through a computer vision algorithm for person detection, a convolutional layer, and several fully connected layers.",
			"TypeIcon"
		],
		[
			"Training Dataset",
			"FitGPT has been trained on fashion brand catalogs, enabling it to infer stylish colors from subjects' body shape and features.",
			"TrainTrackIcon"
		],
		[
			"Inference Speed",
			"Our demo model has been converted to ONNX format to optimize its performance.",
			"InfoIcon"
		]
	]
	
	return (
		<div className="space-y-6">
			<h2 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">Technical Details</h2>
			<div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
				{floaters.map(([titleText, description, iconSrc]) => (
					<Floater
						key={titleText}
						titleText={titleText}
						description={description}
						iconSrc={iconSrc}
						boxBg={true}
					/>
				))}
			</div>
		</div>
	)
}

export default SectionTech;
